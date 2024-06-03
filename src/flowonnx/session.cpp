#include "session.h"
#include "session_p.h"
#include "executionprovider_p.h"

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

    bool Session::open(const fs::path &path, std::string *errorMessage) {
        auto &impl = *_impl;

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
            impl.image = SessionImage::create(path, errorMessage);
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

    Ort::Session createOrtSession(const Ort::Env &env, const std::filesystem::path &modelPath, std::string *errorMessage) {
        try {
            Ort::SessionOptions sessOpt;

            auto ep = Environment::instance()->executionProvider();
            auto deviceIndex = 0;  // TODO: should be a property in Environment

            std::string initEPErrorMsg;
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