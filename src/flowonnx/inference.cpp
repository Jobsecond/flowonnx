#include "inference.h"
#include "inference_p.h"
#include <flowonnx/logger.h>

#include <onnxruntime_cxx_api.h>

#include <sstream>

namespace flowonnx {

    Inference::Inference() : _impl(std::make_unique<Impl>()) {
    }

    Inference::~Inference() = default;

    Inference::Inference(Inference &&other) noexcept {
        std::swap(_impl, other._impl);
    }

    Inference &Inference::operator=(Inference &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        std::swap(_impl, other._impl);
        return *this;
    }

    bool Inference::open(const std::vector<ModelLoadInfo> &models, std::string *errorMessage) {
        auto &impl = *_impl;

        FLOWONNX_DEBUG("Inference - open()");
        std::ostringstream oss;
        bool flag = false;
        for (auto &model : models) {
            std::string loadErrorMessage;
            auto session = Session();
            if (!session.open(model.path, model.preferCpu, &loadErrorMessage)) {
                if (!flag) {
                    oss << "Inference open failed: ";
                    flag = true;
                } else {
                    oss << "; ";
                }
                oss << '[' << model.path << "]: " << loadErrorMessage;
            }
            if (!flag) {
                impl.pathList.push_back(model.path);
                impl.sessionList.push_back(std::move(session));
            }
        }
        if (flag) {
            if (errorMessage) {
                *errorMessage = oss.str();
            }
            impl.pathList.clear();
            impl.sessionList.clear();
            return false;
        }
        return true;
    }

    bool Inference::close() {
        auto &impl = *_impl;
        if (impl.pathList.empty() && impl.sessionList.empty()) {
            return false;
        }
        impl.pathList.clear();
        impl.sessionList.clear();
        return true;
    }

    std::vector<std::string> Inference::inputNames(size_t index) const {
        auto &impl = *_impl;
        if (index >= impl.sessionList.size()) {
            return {};
        }
        return impl.sessionList[index].inputNames();
    }

    std::vector<std::string> Inference::outputNames(size_t index) const {
        auto &impl = *_impl;
        if (index >= impl.sessionList.size()) {
            return {};
        }
        return impl.sessionList[index].outputNames();
    }

    TensorMap Inference::run(std::vector<InferenceData> &inferDataList,
                             std::string *errorMessage) {
        auto &impl = *_impl;
        FLOWONNX_DEBUG("Inference - run()");
        if (!isOpen()) {
            if (errorMessage) {
                *errorMessage = "Inference is not opened!";
            }
            return {};
        }

        if (inferDataList.size() != sessionCount()) {
            if (errorMessage) {
                *errorMessage = "Infer data list length does not match session count!";
            }
            return {};
        }
        std::vector<TensorRefMap> inputMapList(sessionCount());

        TensorMap outMap;
        std::vector<TensorMap> tmpOutTensorList;
        tmpOutTensorList.reserve(sessionCount());
        for (size_t i = 0; i < impl.sessionList.size(); ++i) {
            FLOWONNX_DEBUG("Inference - Processing session %1", i);
            auto &session = impl.sessionList[i];
            auto &inferData = inferDataList[i];
            for (auto &[name, tensor] : inferData.inputData) {
                FLOWONNX_DEBUG("Inference - Session %1: input name \"%2\"", i, name);
                inputMapList[i].emplace(name, tensor);
            }
            std::string sessionRunErrorMessage;
            tmpOutTensorList.emplace_back(session.run(inputMapList[i], &sessionRunErrorMessage));
            auto &out = tmpOutTensorList.back();
            if (out.empty()) {
                if (errorMessage) {
                    *errorMessage = formatTextN("Session %1 run failed: ", i) + sessionRunErrorMessage;
                }
                return {};
            }
            for (auto &name : std::as_const(inferData.outputNames)) {
                if (auto it = out.find(name); it != out.end()) {
                    FLOWONNX_DEBUG("Inference - Session %1 output name \"%2\"", i, name);
                    outMap.emplace(name, std::move(it->second));
                } else {
                    if (errorMessage) {
                        *errorMessage = formatTextN("Could not find \"%1\" from session %2 output names", name, i);
                    }
                    return {};  // error
                }
            }
            for (auto &binding : std::as_const(inferData.bindings)) {
                if (binding.dstIndex >= sessionCount()) {
                    return {}; // error
                }
                if (binding.srcIsInput) {
                    if (auto it = inferData.inputData.find(binding.srcName); it != inferData.inputData.end()) {
                        FLOWONNX_DEBUG("Inference - Binding session %1 input \"%2\" to session %3 input \"%4\"", i,
                                       binding.srcName, binding.dstIndex, binding.dstName);
                        inputMapList[binding.dstIndex].emplace(binding.dstName, it->second);
                    } else {
                        if (errorMessage) {
                            *errorMessage = formatTextN("Bind failed: Could not find \"%1\" from session %2 input!", binding.srcName, i);
                        }
                        return {};  // error
                    }
                } else {
                    if (auto it = out.find(binding.srcName); it != out.end()) {
                        FLOWONNX_DEBUG("Inference - Binding session %1 output \"%2\" to session %3 input \"%4\"", i,
                                       binding.srcName, binding.dstIndex, binding.dstName);
                        inputMapList[binding.dstIndex].emplace(binding.dstName, it->second);
                    } else {
                        if (errorMessage) {
                            *errorMessage = formatTextN("Bind failed: Could not find \"%1\" from session %2 output!", binding.srcName, i);
                        }
                        return {};  // error
                    }
                }
            }
        }

        FLOWONNX_INFO("Inference - inference is successful");
        return outMap;
    }

    bool Inference::terminate() {
        auto &impl = *_impl;
        if (impl.sessionList.empty()) {
            return false;
        }
        for (auto &session : impl.sessionList) {
            session.terminate();
        }
        return true;
    }

    std::filesystem::path Inference::path(size_t index) const {
        auto &impl = *_impl;
        if (index >= impl.pathList.size()) {
            return {};
        }
        return impl.pathList[index];
    }

    bool Inference::isOpen() const {
        auto &impl = *_impl;
        if (impl.sessionList.empty()) {
            return false;
        }
        return std::all_of(impl.sessionList.begin(), impl.sessionList.end(),
                           [](const Session &session) { return session.isOpen(); });
    }

    size_t Inference::sessionCount() const {
        auto &impl = *_impl;
        return impl.sessionList.size();
    }

}