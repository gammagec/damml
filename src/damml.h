#ifndef DAMML_H
#define DAMML_H

namespace damml {

template <typename T>
class Tensor;

class Damml {
public:
  template <typename T>
  [[nodiscard]] auto tensor(T val) const -> Tensor<T>;

  template <typename T>
  [[nodiscard]] auto tensor(const std::vector<size_t>& dims) const -> Tensor<T>;

  template <typename T>
  [[nodiscard]] auto tensor(std::initializer_list<T> data) const -> Tensor<T>;

  template <typename T>
  [[nodiscard]] auto tensor_requires_grad(std::initializer_list<T> data) const -> Tensor<T>;

  template <typename T>
  [[nodiscard]] auto tensor(std::initializer_list<std::initializer_list<T>> data) const -> Tensor<T>;

  template <typename T>
  [[nodiscard]] auto tensor_requires_grad(std::initializer_list<std::initializer_list<T>> data) const -> Tensor<T>;

  template <typename Fn>
  auto with_no_grad(Fn fn) -> void;

  template <typename Fn>
  auto with_enable_grad(Fn fn) -> void;

  template <typename T>
  auto ones(std::vector<size_t> dims) -> Tensor<T>;

private:
  bool grad_enabled_ = true;

  template <typename U>
  friend class Tensor;
};
} // namespace damml

#endif  // DAMML_H
