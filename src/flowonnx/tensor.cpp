#include "tensor.h"

namespace flowonnx {

    template <Tensor::DataType dataType, class T>
    static Tensor createTensorHelper(const T *data, size_t dataSize, const int64_t *shape, size_t shapeSize) {
        Tensor t;
        t.shape.assign(shape, shape + shapeSize);

        auto rawData = reinterpret_cast<const char *>(data);
        t.data.assign(rawData, rawData + dataSize * sizeof(T));

        t.type = dataType;
        return t;
    }

    Tensor Tensor::create(const float *data, size_t dataSize, const int64_t *shape, size_t shapeSize) {
        return createTensorHelper<Float>(data, dataSize, shape, shapeSize);
    }

    Tensor Tensor::create(const int64_t *data, size_t dataSize, const int64_t *shape, size_t shapeSize) {
        return createTensorHelper<Int64>(data, dataSize, shape, shapeSize);
    }

    Tensor Tensor::create(const bool *data, size_t dataSize, const int64_t *shape, size_t shapeSize) {
        return createTensorHelper<Bool>(data, dataSize, shape, shapeSize);
    }

}