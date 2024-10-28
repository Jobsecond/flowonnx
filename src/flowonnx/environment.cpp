#include "environment.h"

#include <stdexcept>

#include "ort_header_fix_p.h"
#include <onnxruntime_cxx_api.h>

#include <loadso/library.h>
#include <loadso/system.h>

#include "format.h"
#include "logger.h"
#include "sessionsystem_p.h"

namespace fs = std::filesystem;

namespace flowonnx {

    static Environment *g_env = nullptr;

    class Environment::Impl {
    public:
        bool load(const fs::path &path, ExecutionProvider ep, std::string *errorMessage) {
            LOG_INFO("flowonnx", "Environment - Loading environment");

            LoadSO::Library tempLib;

            // 1. Load Ort shared library and create handle
            LOG_DEBUG("flowonnx", "Environment - Loading ONNX Runtime shared library");
#ifdef _WIN32
            auto orgLibPath = LoadSO::System::SetLibraryPath(path.parent_path());
#endif
            if (!tempLib.open(path, LoadSO::Library::ResolveAllSymbolsHint)) {
                *errorMessage =
                    formatTextN("%1: Load library failed: %2", path, tempLib.lastError());
                return false;
            }
#ifdef _WIN32
            LoadSO::System::SetLibraryPath(orgLibPath);
#endif

            // 2. Get Ort API getter handle
            LOG_DEBUG("flowonnx", "Environment - Getting ONNX Runtime API handle");
            auto addr = tempLib.resolve("OrtGetApiBase");
            if (!addr) {
                *errorMessage =
                    formatTextN("%1: Get api handle failed: %2", path, tempLib.lastError());
                return false;
            }

            // 3. Check Ort API
            LOG_DEBUG("flowonnx", "Environment - ORT_API_VERSION is " + std::to_string(ORT_API_VERSION));
            auto handle = (OrtApiBase * (ORT_API_CALL *) ()) addr;
            auto apiBase = handle();
            auto api = apiBase->GetApi(ORT_API_VERSION);
            if (!api) {
                *errorMessage = formatTextN("%1: Failed to get api instance");
                return false;
            }

            LOG_DEBUG("flowonnx", "Environment - ONNX Runtime library version is " + std::string(apiBase->GetVersionString()));

            // Successfully get Ort API.
            Ort::InitApi(api);

            std::swap(lib, tempLib);

            loaded = true;
            ortPath = path;
            executionProvider = ep;

            ortApiBase = apiBase;
            ortApi = api;

            LOG_INFO("flowonnx", "Environment - Load successful");
            return true;
        }

        LoadSO::Library lib;

        // Metadata
        bool loaded = false;
        fs::path ortPath;
        ExecutionProvider executionProvider = EP_CPU;
        int deviceIndex = 0;

        // Library data
        void *hLibrary = nullptr;
        const OrtApi *ortApi = nullptr;
        const OrtApiBase *ortApiBase = nullptr;

        SessionSystem sessionSystemInstance;
    };

    Environment::Environment() : _impl(std::make_unique<Impl>()) {
        assert(g_env == nullptr);
        g_env = this;
    }

    Environment::~Environment() {
        g_env = nullptr;
    }

    bool Environment::load(const fs::path &path, ExecutionProvider ep, std::string *errorMessage) {
        auto &impl = *_impl;
        if (impl.loaded) {
            *errorMessage = formatTextN("%1: Library \"%2\" has been loaded", path, impl.ortPath);
            return false;
        }
        return impl.load(path, ep, errorMessage);
    }

    bool Environment::isLoaded() const {
        auto &impl = *_impl;
        return impl.loaded;
    }

    Environment *Environment::instance() {
        return g_env;
    }

    fs::path Environment::runtimePath() const {
        auto &impl = *_impl;
        return impl.ortPath;
    }

    ExecutionProvider Environment::executionProvider() const {
        auto &impl = *_impl;
        return impl.executionProvider;
    }

    int Environment::deviceIndex() const {
        auto &impl = *_impl;
        return impl.deviceIndex;
    }

    void Environment::setDeviceIndex(int deviceIndex) {
        auto &impl = *_impl;
        impl.deviceIndex = deviceIndex;
    }

    std::string Environment::versionString() const {
        auto &impl = *_impl;
        return impl.ortApiBase ? impl.ortApiBase->GetVersionString() : std::string();
    }

}