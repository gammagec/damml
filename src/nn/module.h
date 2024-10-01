#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "tensor/tensor.h"

namespace damml {

template <typename T, typename... Args>
class Module {
public:
  virtual ~Module() = default;

  template <typename... FwdArgs>
  auto operator()(const Tensor<T>& input, const FwdArgs&... args) -> Tensor<T> {
    return forward(input, std::forward<FwdArgs>(args)...);
  }

  virtual auto forward(const Tensor<T>& input, const Args&... args) -> Tensor<T> = 0;

  virtual auto parameters() -> std::vector<Tensor<T>*> = 0;

  virtual auto zero_grad() -> void = 0;
};

} // damml

#endif  // NN_MODULE_H
