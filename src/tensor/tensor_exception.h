#ifndef TENSOR_TENSOR_EXCEPTION_H
#define TENSOR_TENSOR_EXCEPTION_H

#include <exception>
#include <stacktrace>
#include <ranges>

#include "util/util.h"

namespace damml {

class TensorException : public std::exception {
public:
  explicit TensorException(std::string message) : message_(std::move(message)),
                                                  stacktrace_(std::stacktrace::current()) {}

  [[nodiscard]] auto what() const noexcept -> const char* override {
    if (full_message_.empty()) {
      std::ostringstream oss;
      oss << message_ << "\nStacktrace:\n" << stacktrace_;
      full_message_ = oss.str();
    }
    return full_message_.c_str();
  }

  auto stacktrace() const noexcept -> const std::stacktrace& {
    return stacktrace_;
  }

private:
  std::string message_;
  mutable std::string full_message_;
  std::stacktrace stacktrace_;
};

class InvalidRankException final : public TensorException {
public:
  explicit InvalidRankException(size_t rank) : TensorException(std::format("Invalid Rank {}", rank)) {}
};

class InvalidItemAccess final : public TensorException {
public:
  explicit InvalidItemAccess(size_t rank) : TensorException(
    std::format("Invalid Item access, rank must be 0, rank is {}", rank)) {}
};

class OobException final : public TensorException {
public:
  explicit OobException(std::vector<size_t> args, std::vector<size_t> dims) : TensorException(std::format(
    "{} is out of bounds {}",
    args | view_to_string<size_t> | join_with(","),
    dims | view_to_string<size_t> | join_with(","))) {}
};

class DimsMismatchException final : public TensorException {
public:
  explicit DimsMismatchException(std::vector<size_t> dims, std::vector<size_t> other_dims) : TensorException(
    std::format(
      "Dims must match, {} != {}",
      dims | view_to_string<size_t> | join_with(","),
      other_dims | view_to_string<size_t> | join_with(",")
    )) {}
};

} // namespace damml

#endif  // TENSOR_TENSOR_EXCEPTION_H
