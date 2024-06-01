#ifndef EXECUTIONPROVIDER_P_H
#define EXECUTIONPROVIDER_P_H

#include <onnxruntime_cxx_api.h>

namespace flowonnx {
    bool initCUDA(Ort::SessionOptions &options, int deviceIndex, std::string *errorMessage = nullptr);
    bool initDirectML(Ort::SessionOptions &options, int deviceIndex, std::string *errorMessage = nullptr);
}

#endif