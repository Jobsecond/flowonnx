#ifndef INFERENCE_H
#define INFERENCE_H

#include <filesystem>

#include <flowonnx/tensormap.h>

namespace flowonnx {

    struct ModelLoadInfo {
        std::filesystem::path path;
        bool preferCpu = false;
    };

    struct BindingData {
        size_t dstIndex = 0;
        std::string srcName;
        std::string dstName;
        bool srcIsInput = false;
    };

    struct InferenceData {
        TensorMap inputData;
        std::vector<std::string> outputNames;
        std::vector<BindingData> bindings;
    };



    class FLOWONNX_EXPORT Inference {
    public:
        Inference();
        explicit Inference(const std::string &name);
        explicit Inference(std::string &&name);
        ~Inference();

        Inference(Inference &&other) noexcept;
        Inference &operator=(Inference &&other) noexcept;

    public:
        bool open(const std::vector<ModelLoadInfo> &models, std::string *errorMessage = nullptr);
        bool close();

        std::string name() const;
        void setName(const std::string &name);
        void setName(std::string &&name);

        // 获取模型所需的输入和输出名称
        std::vector<std::string> inputNames(size_t index) const;
        std::vector<std::string> outputNames(size_t index) const;

        // 运行推理
        // 如果出错，返回空 TensorMap，并输出错误信息到 errorMessage（可选）
        TensorMap run(std::vector<InferenceData> &inferDataList,
                      std::string *errorMessage = nullptr);

        bool terminate();

        std::filesystem::path path(size_t index) const;
        bool isOpen() const;

        size_t sessionCount() const;
    protected:
        class Impl;
        std::unique_ptr<Impl> _impl;
    };

}

#endif // INFERENCE_H
