#ifndef TENSOR_FUNCTION_HPP
#define TENSOR_FUNCTION_HPP

#include "tensor/function.h"

namespace damml {

template <typename T>
template <typename... Args>
auto BackwardContext<T>::save_for_backward(Args&... args) -> void {
  (save_for_backward_helper(args), ...);
}

template <typename T>
template <typename U>
auto BackwardContext<T>::saved_tensor(const size_t i) -> Tensor<U>* {
  return static_cast<Tensor<U>*>(saved_tensors_[i].get());
}

template <typename T>
auto BackwardContext<T>::needs_input_grad(const size_t i) const -> bool {
  return inputs_[i].needs_grad;
}

template <typename T>
auto BackwardContext<T>::num_inputs() const -> size_t {
  return inputs_.size();
}

template <typename T>
auto BackwardContext<T>::tensor_grad_fn(size_t i) -> GradFn<T>* {
  return &*inputs_[i].grad_fn;
}

template <typename T>
auto BackwardContext<T>::accumulate_output_grad(const Tensor<T>& output_grad) -> void {
  if (output_grad_.has_value()) {
    output_grad_->add_(output_grad);
  } else {
    output_grad_ = output_grad.view();
  }
}

template <typename T>
auto BackwardContext<T>::accumulate_output_grad(size_t i, const Tensor<T>& output_grad) -> void {
  if (inputs_[i].grad_fn != nullptr) {
    inputs_[i].grad_fn->accumulate_output_grad(output_grad);
  } else {
    inputs_[i].grad->add_(output_grad);
  }
}

template <typename T>
auto BackwardContext<T>::output_grad() -> const Tensor<T>& {
  return output_grad_.value();
}

template <typename T>
template <typename ... Args>
auto BackwardContext<T>::set_inputs(Args&... args) -> void {
  (set_input_helper(args), ...);
}

template <typename T>
auto BackwardContext<T>::has_grad_fn(size_t i) -> bool {
  return inputs_[i].grad_fn != nullptr;
}

template <typename T>
template <typename Input>
auto BackwardContext<T>::set_input_helper(Input& input) -> void {
  if (input.requires_grad_ && input.grad_ == nullptr) {
    input.grad_ = std::make_unique<Tensor<T>>(input.context_, input.dims_);
  }
  inputs_.push_back({
    .needs_grad = input.needs_grad(),
    .grad_fn = input.grad_fn_.get(),
    .grad = input.requires_grad_ ?
      std::optional{input.grad_->view()} : std::nullopt
  });
}

template <typename T>
template <typename Input>
auto BackwardContext<T>::save_for_backward_helper(Input& item) {
  saved_tensors_.push_back(item.view_ptr());
}

} // namespace damml

#endif  // TENSOR_FUNCTION_HPP
