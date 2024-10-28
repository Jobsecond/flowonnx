#ifndef FLOWONNX_SESSIONIMAGE_P_H
#define FLOWONNX_SESSIONIMAGE_P_H

#include "ort_header_fix_p.h"
#include <onnxruntime_cxx_api.h>

#include <flowonnx/logger.h>
#include "session_p.h"
#include "sessionsystem_p.h"

namespace flowonnx {

    Ort::Session createOrtSession(const Ort::Env &env, const std::filesystem::path &modelPath, bool preferCpu, std::string *errorMessage = nullptr);

    void loggingFuncOrt(void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message);

    class SessionImage {
    public:
        inline static SessionImage *create(const std::filesystem::path &onnxPath, bool preferCpu, std::string *errorMessage = nullptr);
        inline int ref();
        inline int deref();
    protected:
        inline explicit SessionImage(std::filesystem::path path);
        inline bool init(bool preferCpu, std::string *errorMessage = nullptr);
    public:
        std::filesystem::path path;
        int count;

        std::vector<std::string> inputNames;
        std::vector<std::string> outputNames;

        Ort::Env env;
        Ort::Session session;
    };

    inline SessionImage::SessionImage(std::filesystem::path path)
        : path(std::move(path)), count(1),
          env(ORT_LOGGING_LEVEL_WARNING, "flowonnx", loggingFuncOrt, nullptr),
          session(nullptr) {
    }

    inline int SessionImage::ref() {
        count++;
        LOG_DEBUG("flowonnx", "SessionImage [%1] - ref(), now ref count = %2", path.filename(), count);
        return count;
    }

    inline int SessionImage::deref() {
        count--;
        auto filename = path.filename();
        LOG_DEBUG("flowonnx", "SessionImage [%1] - deref(), now ref count = %2", filename, count);
        if (count == 0) {
            auto &sessionImageMap = SessionSystem::instance()->sessionImageMap;
            auto it = sessionImageMap.find(path);
            if (it != sessionImageMap.end()) {
                LOG_DEBUG("flowonnx", "SessionImage [%1] - removing from session image map", filename);
                sessionImageMap.erase(it);
            }
            LOG_DEBUG("flowonnx", "SessionImage [%1] - delete", filename);
            delete this;
            return 0;
        }
        return count;
    }

    inline bool SessionImage::init(bool preferCpu, std::string *errorMessage) {
        session = createOrtSession(env, path, preferCpu, errorMessage);
        if (session) {
            SessionSystem::instance()->sessionImageMap[path] = this;

            Ort::AllocatorWithDefaultOptions allocator;

            auto inputCount = session.GetInputCount();
            inputNames.reserve(inputCount);
            for (size_t i = 0; i < inputCount; ++i) {
                inputNames.emplace_back(session.GetInputNameAllocated(i, allocator).get());
            }

            auto outputCount = session.GetOutputCount();
            outputNames.reserve(outputCount);
            for (size_t i = 0; i < outputCount; ++i) {
                outputNames.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
            }

            return true;
        }
        return false;
    }

    inline SessionImage *SessionImage::create(const std::filesystem::path &onnxPath, bool preferCpu, std::string *errorMessage) {
        auto filename = onnxPath.filename();
        LOG_DEBUG("flowonnx", "SessionImage [%1] - create", filename);
        auto imagePtr = new SessionImage(onnxPath);
        bool ok = imagePtr->init(preferCpu, errorMessage);
        if (!ok) {
            delete imagePtr;
            LOG_ERROR("flowonnx", "SessionImage [%1] - create failed", filename);
            return nullptr;
        }
        LOG_DEBUG("flowonnx", "SessionImage [%1] - created successfully", filename);
        return imagePtr;
    }
}

#endif // FLOWONNX_SESSIONIMAGE_P_H