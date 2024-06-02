#ifndef SESSION_H
#define SESSION_H

#include <memory>
#include <filesystem>
#include <functional>

#include <flowonnx/flowonnxglobal.h>

namespace flowonnx {

    class FLOWONNX_EXPORT Session {
    public:
        Session();
        ~Session();

        Session(Session &&other) noexcept;
        Session &operator=(Session &&other) noexcept;

    public:
        bool open(const std::filesystem::path &path, std::string *errorMessage);
        bool close();

        std::filesystem::path path() const;
        bool isOpen() const;

    protected:
        class Impl;
        std::unique_ptr<Impl> _impl;
    };

}

#endif // SESSION_H
