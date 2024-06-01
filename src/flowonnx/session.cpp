#include "session.h"
#include "session_p.h"
#include "executionprovider_p.h"

#include <flowonnx/environment.h>

namespace fs = std::filesystem;

namespace flowonnx {

    SessionSystem *SessionSystem::instance() {
        static SessionSystem _instance;
        return &_instance;
    }

    Session::Session() : _impl(std::make_unique<Impl>()) {
    }

    Session::~Session() = default;

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

        fs::path canonicalPath;
        try {
            canonicalPath = fs::canonical(path);
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
            impl.image = new SessionImage(path);
        } else {
            impl.image = it->second;
        }
        bool isImageRefOk;
        impl.image->ref(&isImageRefOk, errorMessage);

        return isImageRefOk;
    }

    bool Session::close() {
        auto &impl = *_impl;
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

    Ort::Session createOrtSession(const Ort::Env &env, const std::filesystem::path &modelPath, std::string *errorMessage) {
        try {
            Ort::SessionOptions sessOpt;

            auto ep = Environment::instance()->executionProvider();
            auto deviceIndex = 0;  // TODO: should be a property in Environment

            std::string initEPErrorMsg;
            switch (ep) {
                case EP_DirectML: {
                    initDirectML(sessOpt, deviceIndex, &initEPErrorMsg);
                    // log warning: "Could not initialize DirectML: {initEPErrorMsg}, use CPU."
                    break;
                }
                case EP_CUDA: {
                    initCUDA(sessOpt, deviceIndex, &initEPErrorMsg);
                    // log warning: "Could not initialize CUDA: {initEPErrorMsg}, use CPU."
                    break;
                }
                default: {
                    // log warning: "Use CPU."
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