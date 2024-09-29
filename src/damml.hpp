#ifndef SRC_DAMML_HPP
#define SRC_DAMML_HPP

#include <vector>

#include "damml.h"
#include "tensor/tensor.h"

namespace damml {

template <typename T>
auto Damml::tensor(T val) const -> Tensor<T> {
  return Tensor<T>(this, val);
}

template <typename T>
auto Damml::tensor(const std::vector<size_t>& dims) const -> Tensor<T> {
  return Tensor<T>(this, dims);
}

template <typename T>
auto Damml::tensor(std::initializer_list<T> data) const -> Tensor<T> {
  return Tensor<T>(this, std::move(data));
}

template <typename T>
auto Damml::tensor_requires_grad(std::initializer_list<T> data) const -> Tensor<T> {
  auto ret = Tensor<T>(this, std::move(data));
  ret.requires_grad_ = true;
  return std::move(ret);
}

template <typename T>
auto Damml::tensor(std::initializer_list<std::initializer_list<T>> data) const -> Tensor<T> {
  return Tensor<T>(this, std::move(data));
}

template <typename T>
auto Damml::tensor_requires_grad(std::initializer_list<std::initializer_list<T>> data) const -> Tensor<T> {
  auto ret = Tensor<T>(this, std::move(data));
  ret.requires_grad_ = true;
  return std::move(ret);
}

template <typename Fn>
auto Damml::with_no_grad(Fn fn) -> void {
  const bool previous = grad_enabled_;
  grad_enabled_ = false;
  fn();
  grad_enabled_ = previous;
}

template <typename Fn>
auto Damml::with_enable_grad(Fn fn) -> void {
  const bool previous = grad_enabled_;
  grad_enabled_ = true;
  fn();
  grad_enabled_ = previous;
}

template <typename T>
auto Damml::ones(std::vector<size_t> dims) -> Tensor<T> {
  auto ret = Tensor<T>(this, dims);
  ret.fill_(static_cast<T>(1));
  return std::move(ret);
}

} // namespace damml

#endif  // SRC_DAMML_HPP
