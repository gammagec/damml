#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include <memory>
#include <vector>

#include "damml.h"
#include "tensor/function.h"

namespace damml {

auto default_strides(const std::vector<size_t>& dims) -> std::vector<int>;

auto vector_product(const std::vector<size_t>& v) -> size_t;

template <std::size_t N, typename... Args>
auto get_nth_value(Args&&... args) -> decltype(auto);

template <typename T>
struct TensorStorage {
  std::vector<T> data;

  explicit TensorStorage(const std::vector<size_t>& dims);

  TensorStorage(std::initializer_list<T> data);
};

template <typename T>
struct MaxResult;

class TensorBase {
};

template <typename T>
class Tensor : public TensorBase {
public:
  Tensor(const Damml* context, T val);

  // Creates a new tensor initialized with all zeros
  Tensor(const Damml* context, const std::vector<size_t>& dims);

  // Creates a new 1d tensor from initializer list
  Tensor(const Damml* context, std::initializer_list<T> values);

  // Creates a new 2d tensor from initializer list
  Tensor(const Damml* context, std::initializer_list<std::initializer_list<T>> values);

  // Creates a view of another tensor's storage
  Tensor(const Damml* context, const std::vector<size_t>& dims, std::shared_ptr<TensorStorage<T>> storage, T* data,
         const std::vector<int>& strides);

  auto operator==(const Tensor& other) const -> bool;
  auto equals_approx(const Tensor& other, T tolerance) const -> bool;

  template <typename... Args>
  auto operator[](Args... args) const -> Tensor;

  template <typename... Args>
  auto operator[](Args... args) -> Tensor;

  auto item() -> T&;

  auto item() const -> const T&;

  auto print(std::ostream& os) const -> void;

  [[nodiscard]] auto dims() const -> const std::vector<size_t>&;

  auto flatten() const -> Tensor;

  auto mm(const Tensor& other) const -> Tensor;

  auto matmul(const Tensor& other) const -> Tensor;

  auto unsqueeze(int dim) const -> Tensor;

  auto squeeze(int dim) const -> Tensor;

  template <typename Fn>
  auto for_each(const Tensor& other, Fn fn) const -> void;

  template <typename Fn>
  auto for_each(Fn fn) const -> void;

  template <typename Fn>
  auto for_each_indexed(Fn fn) const -> void;

  template <typename Predicate>
  auto find_first(const Tensor& other, Predicate pred) const -> std::optional<std::vector<size_t>>;

  auto fill_(T val) -> void;

  auto add_(const Tensor& other) -> void;

  auto add(const Tensor& other) const -> Tensor;

  auto sub(const Tensor& other) const -> Tensor;

  auto t() const -> Tensor;

  auto view() const -> Tensor;
  auto view_ptr() const -> std::unique_ptr<Tensor>;

  template <typename BinaryOp>
  auto transform(const Tensor& other, Tensor& output, BinaryOp op) const -> void;

  template <typename UnaryOp>
  auto transform(Tensor& output, UnaryOp op) const -> void;

  auto neg() const -> Tensor;

  auto operator-() const -> Tensor;

  auto exp() const -> Tensor;

  auto log() const -> Tensor;

  auto tanh() const -> Tensor;

  auto mean() const -> Tensor;

  auto mul(const Tensor& other) const -> Tensor;
  auto operator*(const Tensor& other) const -> Tensor;

  auto div(const Tensor& other) const -> Tensor;
  auto operator/(const Tensor& other) const -> Tensor;

  auto pow(T scalar) const -> Tensor;

  auto expand(const std::vector<int>& dims) const -> Tensor;
  auto expand_to(const Tensor& other) const -> Tensor;

  auto broadcast(const std::vector<int>& dims) const -> Tensor;
  auto broadcast_to(const Tensor& other) const -> Tensor;

  auto sum() const -> Tensor;

  auto sum(int dim, bool keep_dim) const -> Tensor;

  auto max(int dim, bool keep_dim) const -> MaxResult<T>;

  auto backward() const -> void;

  auto backward(const Tensor& output_grad) const -> void;

  auto grad() const -> const Tensor&;

  auto context() const -> const Damml*;

  auto data_size() const -> size_t;

private:
  auto needs_grad() const -> bool;

  const Damml* context_;
  std::vector<size_t> dims_;
  size_t rank_;
  std::shared_ptr<TensorStorage<T>> storage_;
  std::vector<int> strides_;
  T* data_;
  size_t data_size_;
  mutable std::unique_ptr<GradFn<T>> grad_fn_;
  mutable std::unique_ptr<Tensor> grad_;
  bool requires_grad_ = false;

  template <typename U>
  friend class Tensor;

  template <typename U>
  friend class Function;

  friend class Damml;

  template <typename U>
  friend class BackwardContext;

  template <typename U>
  friend auto operator/(U scalar, const Tensor<U>& tensor);

  template <typename U>
  friend auto operator-(U scalar, const Tensor<U>& tensor);
};

template <typename T>
struct MaxResult {
  Tensor<T> values;
  Tensor<size_t> indexes;
};

template <typename T>
auto operator<<(std::ostream& os, const Tensor<T>& tensor) -> std::ostream&;

template <typename T>
auto operator/(T scalar, const Tensor<T>& tensor);

template <typename T>
auto operator-(T scalar, const Tensor<T>& tensor);

} // namespace damml

#endif  // TENSOR_TENSOR_H
