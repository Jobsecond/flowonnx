#ifndef DSINFER_SESSION_P_H
#define DSINFER_SESSION_P_H

#include <map>
#include <utility>

#include <flowonnx/session.h>
#include <flowonnx/logger.h>

#include <onnxruntime_cxx_api.h>

namespace flowonnx {

    Ort::Session createOrtSession(const Ort::Env &env, const std::filesystem::path &modelPath, std::string *errorMessage = nullptr);

    class SessionImage {
    public:
        inline static SessionImage *create(const std::filesystem::path &onnxPath, std::string *errorMessage = nullptr);
        inline int ref();
        inline int deref();
    protected:
        inline explicit SessionImage(std::filesystem::path path);
        inline bool init(std::string *errorMessage = nullptr);
    public:
        std::filesystem::path path;
        int count;

        Ort::Env env;
        Ort::Session session;
    };

    class SessionSystem {
    public:
        std::map<std::filesystem::path, SessionImage *> sessionImageMap;

        static SessionSystem *instance();
    };

    class Session::Impl {
    public:
        SessionImage *image = nullptr;
    };

    inline SessionImage::SessionImage(std::filesystem::path path)
        : path(std::move(path)), count(1),
          env(ORT_LOGGING_LEVEL_WARNING, "flowonnx"),
          session(nullptr) {
    }

    inline int SessionImage::ref() {
        count++;
        FLOWONNX_DEBUG("SessionImage - ref(), now ref count = %1", count);
        return count;
    }

    inline int SessionImage::deref() {
        count--;
        FLOWONNX_DEBUG("SessionImage - deref(), now ref count = %1", count);
        if (count == 0) {
            auto &sessionImageMap = SessionSystem::instance()->sessionImageMap;
            auto it = sessionImageMap.find(path);
            if (it != sessionImageMap.end()) {
                FLOWONNX_DEBUG("SessionImage - removing from session image map");
                sessionImageMap.erase(it);
            }
            FLOWONNX_DEBUG("SessionImage - delete");
            delete this;
            return 0;
        }
        return count;
    }

    inline bool SessionImage::init(std::string *errorMessage) {
        session = createOrtSession(env, path, errorMessage);
        if (session) {
            SessionSystem::instance()->sessionImageMap[path] = this;
            return true;
        }
        return false;
    }

    inline SessionImage *SessionImage::create(const std::filesystem::path &onnxPath, std::string *errorMessage) {
        FLOWONNX_DEBUG("SessionImage - create");
        auto imagePtr = new SessionImage(onnxPath);
        bool ok = imagePtr->init(errorMessage);
        if (!ok) {
            delete imagePtr;
            FLOWONNX_ERROR("SessionImage - create failed");
            return nullptr;
        }
        FLOWONNX_DEBUG("SessionImage - created successfully");
        return imagePtr;
    }

}

#endif // DSINFER_SESSION_P_H
