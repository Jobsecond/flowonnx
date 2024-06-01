#ifndef DSINFER_SESSION_P_H
#define DSINFER_SESSION_P_H

#include <map>
#include <utility>

#include <flowonnx/session.h>

#include <onnxruntime_cxx_api.h>

namespace flowonnx {

    Ort::Session createOrtSession(const Ort::Env &env, const std::filesystem::path &modelPath, std::string *errorMessage = nullptr);

    class SessionImage {
    public:
        inline explicit SessionImage(std::filesystem::path path);
        inline int ref(bool *ok = nullptr, std::string *errorMessage = nullptr);
        inline int deref();

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
        SessionSystem::instance()->sessionImageMap[path] = this;
    }

    inline int SessionImage::ref(bool *ok, std::string *errorMessage) {
        if (count == 0) {
            // init ort session
            session = createOrtSession(env, path, errorMessage);
            bool success = session;
            if (ok) {
                *ok = success;
            }
            if (!success) {
                return 0;
            }
        } else {
            if (ok) {
                *ok = true;
            }
        }
        count++;
        return count;
    }

    inline int SessionImage::deref() {
        count--;
        if (count == 0) {
            SessionSystem::instance()->sessionImageMap.erase(path);
            delete this;
            return 0;
        }
        return count;
    }

}

#endif // DSINFER_SESSION_P_H
