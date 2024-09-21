#include "inference.h"
#include "inference_p.h"
#include <flowonnx/logger.h>
#include <flowonnx/session.h>

#include <onnxruntime_cxx_api.h>

#include <sstream>

namespace flowonnx {

    Inference::Inference() : _impl(std::make_unique<Impl>()) {
    }

    Inference::Inference(const std::string &name) : _impl(std::make_unique<Impl>()) {
        if (_impl) {
            _impl->inferenceName = name;
        }
    }

    Inference::Inference(std::string &&name) : _impl(std::make_unique<Impl>()) {
        if (_impl) {
            _impl->inferenceName = std::move(name);
        }
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

        LOG_DEBUG("[flowonnx] Inference [%1] - open()", impl.inferenceName);
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

    std::string Inference::name() const {
        auto &impl = *_impl;
        return impl.inferenceName;
    }

    void Inference::setName(const std::string &name) {
        auto &impl = *_impl;
        impl.inferenceName = name;
    }

    void Inference::setName(std::string &&name) {
        auto &impl = *_impl;
        impl.inferenceName = std::move(name);
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
        LOG_DEBUG("[flowonnx] Inference [%1] - run()", impl.inferenceName);
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
            LOG_DEBUG("[flowonnx] Inference [%1] - Processing session %2", impl.inferenceName, i);
            auto &session = impl.sessionList[i];
            auto &inferData = inferDataList[i];
            for (auto &[name, tensor] : inferData.inputData) {
                LOG_DEBUG("[flowonnx] Inference [%1] - Session %2: input name \"%3\"", impl.inferenceName, i, name);
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
                    LOG_DEBUG("[flowonnx] Inference [%1] - Session %2 output name \"%3\"", impl.inferenceName, i, name);
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
                        LOG_DEBUG("[flowonnx] Inference [%1] - Binding session %2 input \"%3\" to session %4 input \"%5\"",
                                  impl.inferenceName, i,
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
                        LOG_DEBUG("[flowonnx] Inference [%1] - Binding session %2 output \"%3\" to session %4 input \"%5\"",
                                  impl.inferenceName, i,
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

        LOG_INFO("[flowonnx] Inference [%1] - inference is successful", impl.inferenceName);
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