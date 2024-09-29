#ifndef TENSOR_TENSOR_HPP
#define TENSOR_TENSOR_HPP

#include <vector>
#include <cassert>
#include <cmath>
#include <type_traits>

#include "tensor/tensor.h"
#include "tensor/tensor_exception.h"
#include "damml.h"
#include "util/util.h"
#include "tensor/function.hpp"

namespace damml {

template <std::size_t N, typename... Args>
auto get_nth_value(Args&&... args) -> decltype(auto) {
  return std::get<N>(std::forward_as_tuple(std::forward<Args>(args)...));
}

template <typename T>
TensorStorage<T>::TensorStorage(const std::vector<size_t>& dims) : data(vector_product(dims)) {}

template <typename T>
TensorStorage<T>::TensorStorage(std::initializer_list<T> data) : data(std::move(data)) {}

template <typename T>
Tensor<T>::Tensor(const Damml* context, T val) :
  context_(context),
  dims_({}),
  rank_(0),
  storage_(std::make_shared<TensorStorage<T>>(std::initializer_list<T>{val})),
  strides_({}),
  data_(&storage_->data[0]),
  data_size_(1) {}

template <typename T>
Tensor<T>::Tensor(const Damml* context, const std::vector<size_t>& dims) :
  context_(context),
  dims_(dims),
  rank_(dims.size()),
  storage_(std::make_shared<TensorStorage<T>>(dims)),
  strides_(default_strides(dims)),
  data_(&storage_->data[0]),
  data_size_(vector_product(dims_)) {}

template <typename T>
Tensor<T>::Tensor(const Damml* context, std::initializer_list<T> values) :
  context_(context),
  dims_({values.size()}),
  rank_(1),
  storage_(std::make_shared<TensorStorage<T>>(std::move(values))),
  strides_({1}),
  data_(&storage_->data[0]),
  data_size_(vector_product(dims_)) {}

template <typename T>
Tensor<T>::Tensor(const Damml* context, std::initializer_list<std::initializer_list<T>> values) :
  context_(context),
  dims_({values.size(), values.begin()->size()}),
  rank_(2),
  storage_(std::make_shared<TensorStorage<T>>(dims_)),
  strides_({static_cast<int>(dims_[1]), 1}),
  data_(&storage_->data[0]),
  data_size_(vector_product(dims_)) {
  size_t i = 0;
  for (auto row : values) {
    size_t j = 0;
    if (row.size() != dims_[1]) {
      throw TensorException("Mismatch column sizes in initializer list");
    }
    for (auto col : row) {
      data_[i * strides_[0] + j * strides_[1]] = col;
      ++j;
    }
    ++i;
  }
}

template <typename T>
Tensor<T>::Tensor(const Damml* context,
                  const std::vector<size_t>& dims,
                  std::shared_ptr<TensorStorage<T>> storage,
                  T* data,
                  const std::vector<int>& strides) :
  context_(context),
  dims_(dims),
  rank_(dims.size()),
  storage_(storage),
  strides_(strides),
  data_(data),
  data_size_(vector_product(dims)) {}

template <typename T>
auto Tensor<T>::operator==(const Tensor& other) const -> bool {
  if (dims_ != other.dims_) {
    return false;
  }
  return find_first(other, [](const T& a, const T& b) {
    return a != b;
  }) == std::nullopt;
}

template <typename T>
auto Tensor<T>::equals_approx(const Tensor& other, T tolerance) const -> bool {
  if constexpr (std::is_unsigned_v<T>) {
    throw TensorException("equals approx only works on signed types");
  }
  return find_first(other, [&tolerance](const T& a, const T& b) {
    return std::abs(a - b) > tolerance;
  }) == std::nullopt;
}

template <typename T>
template <typename... Args>
auto Tensor<T>::operator[](Args... args) const -> Tensor {
  if (sizeof...(args) != rank_) {
    throw InvalidRankException(sizeof...(args));
  }
  if constexpr (sizeof...(args) == 0) {
    return Tensor(context_, {}, storage_, data_, {});
  }
  if constexpr (sizeof...(args) == 1) {
    const size_t m = get_nth_value<0>(args...);
    if (m >= dims_[0]) {
      throw OobException({m}, dims_);
    }
    return Tensor(context_, {}, storage_, data_ + m * strides_[0], {});
  }
  if constexpr (sizeof...(args) == 2) {
    const size_t m = get_nth_value<0>(args...);
    const size_t n = get_nth_value<1>(args...);
    if (m >= dims_[0] || n >= dims_[1]) {
      throw OobException({m, n}, dims_);
    }
    return Tensor(context_, {}, storage_, data_ + m * strides_[0] + n * strides_[1], {});
  }
  throw InvalidRankException(sizeof...(args));
}

template <typename T>
template <typename... Args>
auto Tensor<T>::operator[](Args... args) -> Tensor {
  if (sizeof...(args) != rank_) {
    throw InvalidRankException(sizeof...(args));
  }
  if constexpr (sizeof...(args) == 0) {
    return Tensor(context_, {}, storage_, data_, {});
  }
  if constexpr (sizeof...(args) == 1) {
    const size_t m = get_nth_value<0>(args...);
    return Tensor(context_, {}, storage_, data_ + m * strides_[0], {});
  }
  if constexpr (sizeof...(args) == 2) {
    const size_t m = get_nth_value<0>(args...);
    const size_t n = get_nth_value<1>(args...);
    return Tensor(context_, {}, storage_, data_ + m * strides_[0] + n * strides_[1], {});
  }
  throw InvalidRankException(sizeof...(args));
}

template <typename T>
auto Tensor<T>::item() -> T& {
  if (rank_ != 0) {
    throw InvalidItemAccess(rank_);
  }
  return data_[0];
}

template <typename T>
auto Tensor<T>::item() const -> const T& {
  if (rank_ != 0) {
    throw InvalidItemAccess(rank_);
  }
  return data_[0];
}

template <typename T>
auto Tensor<T>::print(std::ostream& os) const -> void {
  os << "Tensor(";
  T* row_ptr = data_;
  switch (rank_) {
  case 0:
    os << data_[0];
    break;
  case 1:
    os << "[";
    for (size_t i = 0; i < dims_[0]; ++i, row_ptr += strides_[0]) {
      if (i != 0) os << ",";
      os << *row_ptr;
    }
    os << "]";
    break;
  case 2:
    os << "[\n";
    for (size_t i = 0; i < dims_[0]; ++i, row_ptr += strides_[0]) {
      T* col_ptr = row_ptr;
      if (i != 0) os << "\n";
      os << "[";
      for (size_t j = 0; j < dims_[1]; ++j, col_ptr += strides_[1]) {
        if (j != 0) os << ",";
        os << *col_ptr;
      }
      os << "]";
    }
    os << "\n]";
    break;
  default:
    throw InvalidRankException(rank_);
  }
  os << ")";
}

template <typename T>
auto Tensor<T>::dims() const -> const std::vector<size_t>& {
  return dims_;
}

template <typename T>
auto Tensor<T>::flatten() const -> Tensor {
  if (rank_ != 2) {
    throw InvalidRankException(rank_);
  }
  auto sz = vector_product(dims_);
  if (strides_[1] == 1 && strides_[0] == dims_[1]) {
    // easy flatten
    return Tensor(context_, {sz}, storage_, data_, {1});
  }
  // create a copy
  Tensor ret(context_, std::vector{sz});
  T* row_ptr = data_;
  size_t k = 0;
  for (size_t i = 0; i < dims_[0]; ++i, row_ptr += strides_[0]) {
    T* col_ptr = row_ptr;
    for (size_t j = 0; j < dims_[1]; ++j, col_ptr += strides_[1], ++k) {
      ret.data_[k] = *col_ptr;
    }
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::mm(const Tensor& other) const -> Tensor {
  if (rank_ != 2) {
    throw InvalidRankException(rank_);
  }
  if (other.rank_ != 2) {
    throw InvalidRankException(other.rank_);
  }
  if (dims_[1] != other.dims_[0]) {
    throw TensorException(std::format("cannot mm {} by {}, 1st cols must match 2nd rows",
                                      dims_ | view_to_string<size_t> | join_with(","),
                                      other.dims_ | view_to_string<size_t> | join_with(",")));
  }

  const size_t m = dims_[0];
  const size_t n = dims_[1];
  const size_t p = other.dims_[1];
  Tensor ret(context_, std::vector{m, p});

  T* a_row_ptr = data_;
  T* ret_row_ptr = ret.data_;
  size_t a_row_stride = strides_[0];
  size_t a_col_stride = strides_[1];
  size_t b_row_stride = other.strides_[0];
  size_t b_col_stride = other.strides_[1];
  size_t ret_row_stride = ret.strides_[0];
  size_t ret_col_stride = ret.strides_[1];
  for (size_t i = 0; i < m; ++i, a_row_ptr += a_row_stride, ret_row_ptr += ret_row_stride) {
    T* a_ptr = a_row_ptr;
    T* b_row_ptr = other.data_;
    for (size_t k = 0; k < n; ++k, b_row_ptr += b_row_stride, a_ptr += a_col_stride) {
      T* b_ptr = b_row_ptr;
      T* ret_ptr = ret_row_ptr;
      for (size_t j = 0; j < p; ++j, b_ptr += b_col_stride, ret_ptr += ret_col_stride) {
        *ret_ptr += *a_ptr * *b_ptr;
      }
    }
  }
  if (ret.rank_ == 2 && (ret.dims_[0] == 1 || ret.dims_[1] == 1)) {
    return ret.flatten();
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::matmul(const Tensor& other) const -> Tensor {
  if (rank_ == 1 && other.rank_ == 1) {
    return unsqueeze(0).mm(other.unsqueeze(1));
  }
  if (rank_ == 2 && other.rank_ == 1) {
    return mm(other.unsqueeze(1));
  }
  if (rank_ == 1 && other.rank_ == 2) {
    return unsqueeze(0).mm(other);
  }
  return mm(other);
}

template <typename T>
auto Tensor<T>::unsqueeze(int dim) const -> Tensor {
  if (dim < 0) {
    dim = rank_ + dim + 1;
  }
  if (rank_ == 0 && dim == 0) {
    return Tensor(context_, {1}, storage_, data_, {1});
  }
  std::vector<size_t> new_dims;
  std::vector<int> new_strides;
  for (size_t i = 0; i < rank_; ++i) {
    if (i == dim) {
      new_dims.push_back(1);
      new_strides.push_back(0);
    }
    new_dims.push_back(dims_[i]);
    new_strides.push_back(strides_[i]);
  }
  if (dim == new_dims.size()) {
    new_dims.push_back(1);
    new_strides.push_back(0);
  }
  return Tensor(context_, new_dims, storage_, data_, new_strides);
}

template <typename T>
auto Tensor<T>::squeeze(int dim) const -> Tensor {
  if (dim < 0) {
    dim = rank_ + dim + 1;
  }
  std::vector<size_t> new_dims;
  std::vector<int> new_strides;
  for (size_t i = 0; i < rank_; ++i) {
    if (i == dim) {
      if (dims_[i] != 1) {
        throw TensorException("Can only squeeze dims of size 1");
      }
      continue;
    }
    new_dims.push_back(dims_[i]);
    new_strides.push_back(strides_[i]);
  }
  return Tensor(context_, new_dims, storage_, data_, new_strides);
}

template <typename T>
template <typename Fn>
auto Tensor<T>::for_each(const Tensor& other, Fn fn) const -> void {
  if (dims_ != other.dims_) {
    throw DimsMismatchException(dims_, other.dims_);
  }
  switch (rank_) {
  case 0:
    fn(data_[0], other.data_[0]);
    break;
  case 1: {
    auto a_stride = strides_[0];
    auto b_stride = other.strides_[0];
    T* a_ptr = data_;
    T* b_ptr = other.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_ptr += a_stride, b_ptr += b_stride) {
      fn(*a_ptr, *b_ptr);
    }
    break;
  }
  case 2: {
    auto a_row_stride = strides_[0];
    auto b_row_stride = other.strides_[0];
    auto a_col_stride = strides_[1];
    auto b_col_stride = other.strides_[1];
    T* a_row_ptr = data_;
    T* b_row_ptr = other.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_row_ptr += a_row_stride, b_row_ptr += b_row_stride) {
      T* a_ptr = a_row_ptr;
      T* b_ptr = b_row_ptr;
      for (size_t j = 0; j < dims_[1]; ++j, a_ptr += a_col_stride, b_ptr += b_col_stride) {
        fn(*a_ptr, *b_ptr);
      }
    }
    break;
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
template <typename Fn>
auto Tensor<T>::for_each(Fn fn) const -> void {
  switch (rank_) {
  case 0:
    fn(data_[0]);
    break;
  case 1: {
    auto a_stride = strides_[0];
    T* a_ptr = data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_ptr += a_stride) {
      fn(*a_ptr);
    }
    break;
  }
  case 2: {
    auto a_row_stride = strides_[0];
    auto a_col_stride = strides_[1];
    T* a_row_ptr = data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_row_ptr += a_row_stride) {
      T* a_ptr = a_row_ptr;
      for (size_t j = 0; j < dims_[1]; ++j, a_ptr += a_col_stride) {
        fn(*a_ptr);
      }
    }
    break;
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
template <typename Fn>
auto Tensor<T>::for_each_indexed(Fn fn) const -> void {
  switch (rank_) {
  case 1: {
    if constexpr (function_argument_count_v<Fn> == 2) {
      auto a_stride = strides_[0];
      T* a_ptr = data_;
      for (size_t i = 0; i < dims_[0]; ++i, a_ptr += a_stride) {
        fn(i, *a_ptr);
      }
    }
    else {
      throw TensorException("for_each_indexed on 1d tensor requires 2 args");
    }
    break;
  }
  case 2: {
    if constexpr (function_argument_count_v<Fn> == 3) {
      auto a_row_stride = strides_[0];
      auto a_col_stride = strides_[1];
      T* a_row_ptr = data_;
      for (size_t i = 0; i < dims_[0]; ++i, a_row_ptr += a_row_stride) {
        T* a_ptr = a_row_ptr;
        for (size_t j = 0; j < dims_[1]; ++j, a_ptr += a_col_stride) {
          fn(i, j, *a_ptr);
        }
      }
    }
    else {
      throw TensorException("for_each_indexed on 2d tensor requires 3 args");
    }
    break;
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
template <typename Predicate>
auto Tensor<T>::find_first(const Tensor& other, Predicate pred) const -> std::optional<std::vector<size_t>> {
  if (dims_ != other.dims_) {
    throw DimsMismatchException(dims_, other.dims_);
  }
  switch (rank_) {
  case 0:
    if (pred(data_[0], other.data_[0])) {
      return std::vector<size_t>{};
    }
    break;
  case 1: {
    auto a_stride = strides_[0];
    auto b_stride = other.strides_[0];
    T* a_ptr = data_;
    T* b_ptr = other.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_ptr += a_stride, b_ptr += b_stride) {
      if (pred(*a_ptr, *b_ptr)) {
        return std::vector{i};
      }
    }
    break;
  }
  case 2: {
    auto a_row_stride = strides_[0];
    auto b_row_stride = other.strides_[0];
    auto a_col_stride = strides_[1];
    auto b_col_stride = other.strides_[1];
    T* a_row_ptr = data_;
    T* b_row_ptr = other.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_row_ptr += a_row_stride, b_row_ptr += b_row_stride) {
      T* a_ptr = a_row_ptr;
      T* b_ptr = b_row_ptr;
      for (size_t j = 0; j < dims_[1]; ++j, a_ptr += a_col_stride, b_ptr += b_col_stride) {
        if (pred(*a_ptr, *b_ptr)) {
          return std::vector{i, j};
        }
      }
    }
    break;
  }
  default:
    throw InvalidRankException(rank_);
  }
  return std::nullopt;
}

template <typename T>
auto Tensor<T>::fill_(T val) -> void {
  transform(*this, [&val](const T&) {
    return val;
  });
}

template <typename T>
auto Tensor<T>::t() const -> Tensor {
  if (rank_ == 1) {
    return Tensor(context_, dims_, storage_, data_, strides_);
  }
  if (rank_ == 2) {
    return Tensor(context_, dims_, storage_, data_, {strides_[1], strides_[0]});
  }
  throw InvalidRankException(rank_);
}

template <typename T>
auto Tensor<T>::view() const -> Tensor {
  return Tensor(context_, dims_, storage_, data_, strides_);
}

template <typename T>
auto Tensor<T>::view_ptr() const -> std::unique_ptr<Tensor> {
  return std::make_unique<Tensor>(context_, dims_, storage_, data_, strides_);
}

template <typename T>
template <typename BinaryOp>
auto Tensor<T>::transform(const Tensor& other, Tensor& output, BinaryOp op) const -> void {
  if (dims_ != other.dims_) {
    throw DimsMismatchException(dims_, other.dims_);
  }
  if (dims_ != output.dims_) {
    throw DimsMismatchException(dims_, output.dims_);
  }
  switch (rank_) {
  case 0:
    output.data_[0] = op(data_[0], other.data_[0]);
    break;
  case 1: {
    auto a_stride = strides_[0];
    auto b_stride = other.strides_[0];
    auto out_stride = output.strides_[0];
    T* a_ptr = data_;
    T* b_ptr = other.data_;
    T* out_ptr = output.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_ptr += a_stride, b_ptr += b_stride, out_ptr += out_stride) {
      *out_ptr = op(*a_ptr, *b_ptr);
    }
    break;
  }
  case 2: {
    auto a_row_stride = strides_[0];
    auto b_row_stride = other.strides_[0];
    auto a_col_stride = strides_[1];
    auto b_col_stride = other.strides_[1];
    auto out_row_stride = output.strides_[0];
    auto out_col_stride = output.strides_[1];
    T* a_row_ptr = data_;
    T* b_row_ptr = other.data_;
    T* out_row_ptr = output.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_row_ptr += a_row_stride, b_row_ptr += b_row_stride, out_row_ptr +=
         out_row_stride) {
      T* a_ptr = a_row_ptr;
      T* b_ptr = b_row_ptr;
      T* out_ptr = out_row_ptr;
      for (size_t j = 0; j < dims_[1]; ++j, a_ptr += a_col_stride, b_ptr += b_col_stride, out_ptr += out_col_stride) {
        *out_ptr = op(*a_ptr, *b_ptr);
      }
    }
    break;
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
template <typename UnaryOp>
auto Tensor<T>::transform(Tensor& output, UnaryOp op) const -> void {
  if (dims_ != output.dims_) {
    throw DimsMismatchException(dims_, output.dims_);
  }
  switch (rank_) {
  case 0:
    output.data_[0] = op(data_[0]);
    break;
  case 1: {
    auto a_stride = strides_[0];
    auto out_stride = output.strides_[0];
    T* a_ptr = data_;
    T* out_ptr = output.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_ptr += a_stride, out_ptr += out_stride) {
      *out_ptr = op(*a_ptr);
    }
    break;
  }
  case 2: {
    auto a_row_stride = strides_[0];
    auto a_col_stride = strides_[1];
    auto out_row_stride = output.strides_[0];
    auto out_col_stride = output.strides_[1];
    T* a_row_ptr = data_;
    T* out_row_ptr = output.data_;
    for (size_t i = 0; i < dims_[0]; ++i, a_row_ptr += a_row_stride, out_row_ptr +=
         out_row_stride) {
      T* a_ptr = a_row_ptr;
      T* out_ptr = out_row_ptr;
      for (size_t j = 0; j < dims_[1]; ++j, a_ptr += a_col_stride, out_ptr += out_col_stride) {
        *out_ptr = op(*a_ptr);
      }
    }
    break;
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
auto Tensor<T>::neg() const -> Tensor {
  Tensor ret(context_, dims_);
  transform(ret, [](const T& val) {
    return -val;
  });
  if (needs_grad()) {
    ret.grad_fn_ = std::make_unique<NegBackward<T>>(*this);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::operator-() const -> Tensor {
  return neg();
}

template <typename T>
auto Tensor<T>::exp() const -> Tensor {
  Tensor ret(context_, dims_);
  transform(ret, [](const T& val) {
    return std::exp(val);
  });
  if (needs_grad()) {
    ret.grad_fn_ = std::make_unique<ExpBackward<T>>(*this, ret);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::log() const -> Tensor {
  Tensor ret(context_, dims_);
  transform(ret, [](const T& val) {
    return std::log(val);
  });
  if (needs_grad()) {
    ret.grad_fn_ = std::make_unique<LogBackward<T>>(*this);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::tanh() const -> Tensor {
  Tensor ret(context_, dims_);
  transform(ret, [](const T& val) {
    return std::tanh(val);
  });
  if (needs_grad()) {
    ret.grad_fn_ = std::make_unique<TanhBackward<T>>(*this, ret);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::mean() const -> Tensor {
  T sum = 0;
  for_each([&sum](const T& val) {
    sum += val;
  });
  auto ret = Tensor(context_, sum / static_cast<T>(data_size_));
  if (needs_grad()) {
    ret.grad_fn_ = std::make_unique<MeanBackward<T>>(*this);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::sub(const Tensor& other) const -> Tensor {
  if (dims_ != other.dims_) {
    if (rank_ < other.rank_) {
      return broadcast_to(other).sub(other);
    }
    if (rank_ > other.rank_) {
      return sub(other.broadcast_to(*this));
    }
  }
  Tensor ret(context_, dims_);
  transform(other, ret, std::minus());
  if (needs_grad() || other.needs_grad()) {
    ret.grad_fn_ = std::make_unique<SubBackward<T>>(*this, other);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::add(const Tensor& other) const -> Tensor {
  Tensor ret(context_, dims_);
  transform(other, ret, std::plus());
  if (needs_grad() || other.needs_grad()) {
    ret.grad_fn_ = std::make_unique<AddBackward<T>>(*this, other);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::add_(const Tensor& other) -> void {
  transform(other, *this, std::plus());
}

template <typename T>
auto Tensor<T>::mul(const Tensor& other) const -> Tensor {
  if (dims_ != other.dims_) {
    if (rank_ < other.rank_) {
      return broadcast_to(other).mul(other);
    }
    if (rank_ > other.rank_) {
      return mul(other.broadcast_to(*this));
    }
  }
  Tensor ret(context_, dims_);
  transform(other, ret, std::multiplies());
  if (needs_grad() || other.needs_grad()) {
    ret.grad_fn_ = std::make_unique<MulBackward<T>>(*this, other);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::operator*(const Tensor& other) const -> Tensor {
  return mul(other);
}

template <typename T>
auto Tensor<T>::div(const Tensor& other) const -> Tensor {
  if (dims_ != other.dims_) {
    if (rank_ < other.rank_) {
      return broadcast_to(other).div(other);
    }
    if (rank_ > other.rank_) {
      return div(other.broadcast_to(*this));
    }
  }
  Tensor ret(context_, dims_);
  transform(other, ret, std::divides());
  if (needs_grad() || other.needs_grad()) {
    ret.grad_fn_ = std::make_unique<DivBackward<T>>(*this, other);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::operator/(const Tensor& other) const -> Tensor {
  return div(other);
}

template <typename T>
auto Tensor<T>::pow(T scalar) const -> Tensor {
  Tensor ret(context_, dims_);
  transform(ret, [&scalar](const T& val) {
    return std::pow(val, scalar);
  });
  if (needs_grad()) {
    ret.grad_fn_ = std::make_unique<PowBackward<T>>(*this, scalar);
  }
  return std::move(ret);
}

template <typename T>
auto Tensor<T>::expand(const std::vector<int>& dims) const -> Tensor {
  assert(rank_ == dims.size());
  std::vector<size_t> new_dims;
  std::vector<int> new_strides;
  for (size_t i = 0; i < rank_; ++i) {
    if (dims[i] == -1 || dims[i] == dims_[i]) {
      new_dims.push_back(dims_[i]);
      new_strides.push_back(strides_[i]);
    }
    else if (dims_[i] == 1 && dims[i] > 0) {
      new_dims.push_back(dims[i]);
      new_strides.push_back(0);
    }
    else {
      throw TensorException("invalid expand");
    }
  }
  return Tensor(context_, new_dims, storage_, data_, new_strides);
}

template <typename T>
auto Tensor<T>::expand_to(const Tensor& other) const -> Tensor {
  return expand(other.dims_ |
    std::views::transform([](const size_t val) {
      return static_cast<int>(val);
    }) | std::ranges::to<std::vector<int>>());
}

template <typename T>
auto Tensor<T>::broadcast(const std::vector<int>& dims) const -> Tensor {
  if (rank_ > dims.size()) {
    throw TensorException("Cannot broadcast to a smaller rank");
  }
  if (rank_ < dims.size()) {
    return unsqueeze(0).broadcast(dims);
  }
  // ranks equal
  return expand(dims);
}

template <typename T>
auto Tensor<T>::broadcast_to(const Tensor& other) const -> Tensor {
  return broadcast(other.dims_ |
    std::views::transform([](const size_t val) {
      return static_cast<int>(val);
    }) | std::ranges::to<std::vector<int>>());
}

template <typename T>
auto Tensor<T>::sum() const -> Tensor {
  T sum = 0;
  for_each([&sum](const T& val) {
    sum += val;
  });
  return Tensor(sum);
}

template <typename T>
auto Tensor<T>::sum(int dim, const bool keep_dim) const -> Tensor {
  if (dim < 0) {
    dim = rank_ + dim;
  }
  switch (rank_) {
  case 1: {
    if (dim != 0) {
      throw TensorException("invalid dim, rank 1 tensor can only be 0");
    }
    Tensor ret(context_, {static_cast<T>(0)});
    for_each([&ret](const T& val) {
      ret.data_[0] += val;
    });
    if (keep_dim) {
      return std::move(ret);
    }
    return ret[0];
  }
  case 2: {
    if (dim == 0) { // sum along rows
      Tensor ret(context_, std::vector{1, dims_[1]});
      auto row_ptr = data_;
      auto row_stride = strides_[0];
      auto col_stride = strides_[1];
      for (size_t i = 0; i < dims_[0]; ++i, row_ptr += row_stride) {
        T* col_ptr = row_ptr;
        T* ret_ptr = ret.data_;
        for (size_t j = 0; j < dims_[1]; ++j, col_ptr += col_stride, ++ret_ptr) {
          *ret_ptr += *col_ptr;
        }
      }
      if (keep_dim) return std::move(ret);
      return ret.squeeze(0);
    }
    if (dim == 1) { // sum along cols
      Tensor ret(context_, std::vector<size_t>{dims_[0], 1});
      auto row_ptr = data_;
      auto row_stride = strides_[0];
      auto col_stride = strides_[1];
      T* ret_ptr = ret.data_;
      for (size_t i = 0; i < dims_[0]; ++i, row_ptr += row_stride, ++ret_ptr) {
        T* col_ptr = row_ptr;
        for (size_t j = 0; j < dims_[1]; ++j, col_ptr += col_stride) {
          *ret_ptr += *col_ptr;
        }
      }
      if (keep_dim) return std::move(ret);
      return ret.squeeze(1);
    }
    throw TensorException("Invalid dim. Rank 2 tensor can only be 0 or 1");
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
auto Tensor<T>::max(int dim, const bool keep_dim) const -> MaxResult<T> {
  if (dim < 0) {
    dim = rank_ + dim;
  }
  switch (rank_) {
  case 1: {
    if (dim != 0) {
      throw TensorException("invalid dim, rank 1 tensor can only be 0");
    }
    Tensor ret(context_, {static_cast<T>(0)});
    size_t index = 0;
    // get max with index
    for_each_indexed([&ret, &index](const size_t i, const T& val) {
      if (i == 0 || val > ret.data_[0]) {
        ret.data_[0] = val;
        index = i;
      }
    });
    if (keep_dim) {
      return MaxResult<T>{
        .values = std::move(ret),
        .indexes = Tensor<size_t>(context_, index)
      };
    }
    return MaxResult<T>{
      .values = Tensor(context_, ret.data_[0]),
      .indexes = Tensor<size_t>(context_, index)
    };
  }
  case 2: {
    if (dim == 0) { // max along rows
      Tensor ret(context_, std::vector{1, dims_[1]});
      Tensor<size_t> indexes(context_, std::vector{dims_[1]});
      for_each_indexed([&ret, &indexes](const size_t i, const size_t j, const T& val) {
        if (i == 0 || val > ret.data_[j]) {
          ret.data_[j] = val;
          indexes.data_[j] = i;
        }
      });
      if (keep_dim) {
        return MaxResult<T>{
          .values = std::move(ret),
          .indexes = std::move(indexes)
        };
      }
      return MaxResult<T>{
        .values = ret.squeeze(0),
        .indexes = std::move(indexes)
      };
    }
    if (dim == 1) { // max along cols
      Tensor ret(context_, std::vector<size_t>{dims_[0], 1});
      Tensor<size_t> indexes(context_, std::vector<size_t>{dims_[0]});
      for_each_indexed([&ret, &indexes](const size_t i, const size_t j, const T& val) {
        if (j == 0 || val > ret.data_[i]) {
          ret.data_[i] = val;
          indexes.data_[i] = j;
        }
      });
      if (keep_dim) {
        return MaxResult<T>{
          .values = std::move(ret),
          .indexes = std::move(indexes)
        };
      }
      return MaxResult<T>{
        .values = ret.squeeze(1),
        .indexes = std::move(indexes)
      };
    }
  }
  default:
    throw InvalidRankException(rank_);
  }
}

template <typename T>
auto Tensor<T>::backward() const -> void {
  if (rank_ != 0) {
    throw TensorException("Can only do backward on a scalar value");
  }
  return backward(Tensor<T>(context_, 1));
}

template <typename T>
auto Tensor<T>::backward(const Tensor& output_grad) const -> void {
  if (grad_fn_ == nullptr) {
    throw TensorException("Can only do backward when a child has requires_grad");
  }
  grad_fn_->accumulate_output_grad(output_grad);
  std::set<GradFn<T>*> visited;
  std::list<GradFn<T>*> topo;
  grad_fn_->build_topo(visited, topo);
  topo.reverse();
  for (auto grad_fn : topo) {
    grad_fn->backward();
  }
}

template <typename T>
auto Tensor<T>::grad() const -> const Tensor& {
  return *grad_;
}

template <typename T>
auto Tensor<T>::context() const -> const Damml* {
  return context_;
}

template <typename T>
auto Tensor<T>::data_size() const -> size_t {
  return data_size_;
}

template <typename T>
auto Tensor<T>::needs_grad() const -> bool {
  return context_->grad_enabled_ && (grad_fn_ != nullptr || requires_grad_);
}

template <typename T>
auto operator<<(std::ostream& os, const Tensor<T>& tensor) -> std::ostream& {
  tensor.print(os);
  return os;
}

template <typename T>
auto operator-(T scalar, const Tensor<T>& tensor) {
  return Tensor(tensor.context_, scalar).sub(tensor);
}

template <typename T>
auto operator/(T scalar, const Tensor<T>& tensor) {
  return Tensor(tensor.context_, scalar).div(tensor);
}

} // namespace damml

#endif  // TENSOR_TENSOR_HPP
