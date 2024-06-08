#ifndef SESSION_H
#define SESSION_H

#include <map>
#include <memory>
#include <filesystem>
#include <functional>

#include <flowonnx/flowonnxglobal.h>
#include <flowonnx/tensor.h>

namespace flowonnx {

    // { 名称: 张量 }
    using TensorMap = std::map<std::string, Tensor>;

    class FLOWONNX_EXPORT Session {
    public:
        Session();
        ~Session();

        Session(Session &&other) noexcept;
        Session &operator=(Session &&other) noexcept;

    public:
        bool open(const std::filesystem::path &path, bool forceOnCpu, std::string *errorMessage);
        bool close();

        // 获取模型所需的输入和输出名称
        std::vector<std::string> inputNames() const;
        std::vector<std::string> outputNames() const;

        // 运行推理
        // 如果出错，返回空 TensorMap，并输出错误信息到 errorMessage（可选）
        TensorMap run(TensorMap &inputTensorMap, std::string *errorMessage = nullptr);

        void terminate();

        std::filesystem::path path() const;
        bool isOpen() const;

    protected:
        class Impl;
        std::unique_ptr<Impl> _impl;
    };

}

#endif // SESSION_H
