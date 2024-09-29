#include "tensor/tensor.h"

#include <vector>
#include <numeric>

#include "tensor/tensor_exception.h"

namespace damml {

auto vector_product(const std::vector<size_t>& v) -> size_t {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<>());
}

auto default_strides(const std::vector<size_t>& dims) -> std::vector<int> {
  std::vector<int> strides(dims.size());
  switch (dims.size()) {
  case 0:
    // No strides;
    break;
  case 1:
    strides[0] = 1;
    break;
  case 2:
    strides[0] = static_cast<int>(dims[1]);
    strides[1] = 1;
    break;
  default:
    throw InvalidRankException(dims.size());
  }
  return std::move(strides);
}

} // namespace damml