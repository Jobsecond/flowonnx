#ifndef INFERENCE_P_H
#define INFERENCE_P_H

#include <flowonnx/session.h>
#include <flowonnx/inference.h>

namespace flowonnx {

    class Inference::Impl {
    public:
        std::vector<std::filesystem::path> pathList;
        std::vector<Session> sessionList;
    };

}

#endif // INFERENCE_P_H
