#ifndef TENSORMAP_H
#define TENSORMAP_H

#include <map>
#include <string>
#include <type_traits>

#include <flowonnx/tensor.h>

namespace flowonnx {
    using TensorMap = std::map<std::string, Tensor>;
    using TensorRefMap = std::map<std::string, std::reference_wrapper<Tensor>>;
}
#endif // TENSORMAP_H