#ifndef FLOWONNX_SESSION_P_H
#define FLOWONNX_SESSION_P_H

#include "ort_header_fix_p.h"
#include <onnxruntime_cxx_api.h>

#include <flowonnx/session.h>
#include "sessionsystem_p.h"

namespace flowonnx {

    class SessionImage;

    class Session::Impl {
    public:
        SessionImage *image = nullptr;
        Ort::RunOptions runOptions;
    };



}

#endif // FLOWONNX_SESSION_P_H
