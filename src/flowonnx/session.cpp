#include "session.h"
#include "session_p.h"
#include "executionprovider_p.h"

#include <sstream>
#include <unordered_set>
#include <flowonnx/environment.h>
#include <flowonnx/logger.h>

namespace fs = std::filesystem;

namespace flowonnx {

    SessionSystem *SessionSystem::instance() {
        static SessionSystem _instance;
        return &_instance;
    }

    Session::Session() : _impl(std::make_unique<Impl>()) {
    }

    Session::~Session() {
        close();
    }

    Session::Session(Session &&other) noexcept {
        std::swap(_impl, other._impl);
    }

    Session &Session::operator=(Session &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        std::swap(_impl, other._impl);
        return *this;
    }

    bool Session::open(const fs::path &path, bool forceOnCpu, std::string *errorMessage) {
        auto &impl = *_impl;
        // TODO: If the same session is already opened before, forceOnCpu will have no effect
        //       due to SessionSystem will return the existing SessionImage instead creating a new one.
        //       Should this be the desired behavior, or it needs to be fixed?

        FLOWONNX_DEBUG("Session - Try open " + path.string());
        fs::path canonicalPath;
        try {
            canonicalPath = fs::canonical(path);
            FLOWONNX_DEBUG("Session - The canonical path is " + canonicalPath.string());
        } catch (const std::exception &e) {
            if (errorMessage) {
                *errorMessage = e.what();
            }
            return false;
        }

        if (!fs::is_regular_file(canonicalPath)) {
            if (errorMessage) {
                *errorMessage = "Not a regular file";
            }
            return false;
        }

        auto mgr = SessionSystem::instance();
        auto it = mgr->sessionImageMap.find(canonicalPath);
        if (it == mgr->sessionImageMap.end()) {
            FLOWONNX_DEBUG("Session - The session image does not exist. Creating a new one...");
            impl.image = SessionImage::create(path, forceOnCpu, errorMessage);
        } else {
            FLOWONNX_DEBUG("Session - The session image already exists. Increasing the reference count...");
            impl.image = it->second;
            impl.image->ref();
        }

        return impl.image != nullptr;
    }

    bool Session::close() {
        auto &impl = *_impl;
        FLOWONNX_DEBUG("Session - close");
        if (!impl.image)
            return false;

        if (impl.image->deref() == 0) {
            impl.image = nullptr;
        }
        return true;
    }

    fs::path Session::path() const {
        auto &impl = *_impl;
        return impl.image ? impl.image->path : fs::path();
    }

    bool Session::isOpen() const {
        auto &impl = *_impl;
        return impl.image != nullptr;
    }

    template<typename T>
    inline Ort::Value createOrtValueHelper(Tensor &tensor, const Ort::MemoryInfo &memoryInfo) {
        T *dataBuffer;
        auto dataSize = tensor.getDataBuffer<T>(&dataBuffer);
        return Ort::Value::CreateTensor<T>(memoryInfo, dataBuffer, dataSize, tensor.shape.data(), tensor.shape.size());
    }

    TensorMap Session::run(TensorMap &inputTensorMap, std::string *errorMessage) {
        // Here we don't use const TensorMap & because Ort::Value::CreateTensor requires a non-const buffer
        auto &impl = *_impl;
        FLOWONNX_DEBUG("Session - Running inference");
        if (!impl.image) {
            if (errorMessage) {
                *errorMessage = "Session is not open";
            }
            return {};
        }

        if (inputTensorMap.empty()) {
            if (errorMessage) {
                *errorMessage = "Input map is empty";
            }
            return {};
        }

        const auto &requiredInputNames = impl.image->inputNames;
        std::ostringstream msgStream;

        // Check for missing and extra input names. If found, return empty map and the error message.
        {
            bool flagMissing = false;
            // Check for missing input names

            for (const auto &requiredInputName: requiredInputNames) {
                if (inputTensorMap.find(requiredInputName) == inputTensorMap.end()) {
                    if (flagMissing) {
                        // It isn't the first missing input name. Append a comma separator.
                        msgStream << ',' << ' ';
                    } else {
                        // It's the first missing input name. Append the message intro.
                        msgStream << "Missing input name(s): ";
                        flagMissing = true;
                    }
                    msgStream << '"' << requiredInputName << '"';
                }
            }

            // Check for extra input names
            bool flagExtra = false;
            std::unordered_set<std::string> requiredSet(requiredInputNames.begin(), requiredInputNames.end());
            for (auto &it: std::as_const(inputTensorMap)) {
                auto &actualInputName = it.first;
                if (requiredSet.find(actualInputName) == requiredSet.end()) {
                    if (flagExtra) {
                        msgStream << ',' << ' ';
                    } else {
                        if (flagMissing) {
                            msgStream << ';' << ' ';
                        }
                        msgStream << "Extra input names(s): ";
                        flagExtra = true;
                    }
                    msgStream << '"' << actualInputName << '"';
                }
            }

            if (flagMissing || flagExtra) {
                if (errorMessage) {
                    *errorMessage = msgStream.str();
                }
                return {};
            }
        }

        try {
            auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::IoBinding binding(impl.image->session);
            for (auto &[name, tensor]: inputTensorMap) {
                switch (tensor.type) {
                    case Tensor::Float:
                        binding.BindInput(name.c_str(), createOrtValueHelper<float>(tensor, memInfo));
                        break;
                    case Tensor::Int64:
                        binding.BindInput(name.c_str(), createOrtValueHelper<int64_t>(tensor, memInfo));
                        break;
                    case Tensor::Bool:
                        binding.BindInput(name.c_str(), createOrtValueHelper<bool>(tensor, memInfo));
                        break;
                    default:
                        if (errorMessage) {
                            *errorMessage = formatTextN("Tensor data type for \"%1\" is not implemented!", name);
                        }
                        return {};
                }
            }

            const auto &outputNames = impl.image->outputNames;
            for (const auto &name: outputNames) {
                binding.BindOutput(name.c_str(), memInfo);
            }

            impl.runOptions.UnsetTerminate();
            impl.image->session.Run(impl.runOptions, binding);

            TensorMap outTensorMap;
            auto outputValues = binding.GetOutputValues();
            for (size_t i = 0; i < outputValues.size(); ++i) {
                const auto &name = outputNames[i];
                const auto &value = outputValues[i];
                auto tsInfo = value.GetTensorTypeAndShapeInfo();
                auto shape = tsInfo.GetShape();
                auto size = tsInfo.GetElementCount();
                auto type = tsInfo.GetElementType();
                switch (type) {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                        outTensorMap.emplace(name, Tensor::create(value.GetTensorData<float>(), size, shape.data(), shape.size()));
                        break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                        outTensorMap.emplace(name, Tensor::create(value.GetTensorData<int64_t>(), size, shape.data(), shape.size()));
                        break;
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                        outTensorMap.emplace(name, Tensor::create(value.GetTensorData<bool>(), size, shape.data(), shape.size()));
                        break;
                    default:
                        if (errorMessage) {
                            *errorMessage = formatTextN("Tensor data type for \"%1\" is not implemented!", name);
                        }
                        return {};
                }
            }
            return outTensorMap;
        } catch (const Ort::Exception &err) {
            if (errorMessage) {
                *errorMessage = err.what();
            }
        }
        return {};
    }

    std::vector<std::string> Session::inputNames() const {
        auto &impl = *_impl;
        if (!impl.image) {
            return {};
        }
        return impl.image->inputNames;
    }

    std::vector<std::string> Session::outputNames() const {
        auto &impl = *_impl;
        if (!impl.image) {
            return {};
        }
        return impl.image->outputNames;
    }

    void Session::terminate() {
        auto &impl = *_impl;
        impl.runOptions.SetTerminate();
    }

    Ort::Session createOrtSession(const Ort::Env &env, const std::filesystem::path &modelPath, bool forceOnCpu, std::string *errorMessage) {
        try {
            Ort::SessionOptions sessOpt;

            auto ep = Environment::instance()->executionProvider();
            auto deviceIndex = 0;  // TODO: should be a property in Environment

            std::string initEPErrorMsg;
            if (!forceOnCpu) {
                switch (ep) {
                    case EP_DirectML: {
                        if (!initDirectML(sessOpt, deviceIndex, &initEPErrorMsg)) {
                            // log warning: "Could not initialize DirectML: {initEPErrorMsg}, use CPU."
                            FLOWONNX_WARNING("Could not initialize DirectML: %1, use CPU.", initEPErrorMsg);
                        } else {
                            FLOWONNX_INFO("Use DirectML.");
                        }
                        break;
                    }
                    case EP_CUDA: {
                        if (!initCUDA(sessOpt, deviceIndex, &initEPErrorMsg)) {
                            // log warning: "Could not initialize CUDA: {initEPErrorMsg}, use CPU."
                            FLOWONNX_WARNING("Could not initialize CUDA: %1, use CPU.", initEPErrorMsg);
                        } else {
                            FLOWONNX_INFO("Use CUDA.");
                        }
                        break;
                    }
                    default: {
                        // log info: "Use CPU."
                        FLOWONNX_INFO("Use CPU.");
                        break;
                    }
                }
            } else {
                FLOWONNX_INFO("Use CPU.");
            }

#ifdef _WIN32
            auto pathStr = modelPath.wstring();
#else
            auto pathStr = modelPath.string();
#endif
            return Ort::Session{ env, pathStr.c_str(), sessOpt };
        } catch (const Ort::Exception &e) {
            if (errorMessage) {
                *errorMessage = e.what();
            }
        }
        return Ort::Session{ nullptr };
    }
}