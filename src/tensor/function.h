#ifndef TENSOR_FUNCTION_H
#define TENSOR_FUNCTION_H

#include <set>
#include <list>

namespace damml {

template <typename T>
class GradFn;

class TensorBase;

template <typename T>
class BackwardContext {
public:
  template <typename... Args>
  auto save_for_backward(Args&... args) -> void;

  template <typename U>
  auto saved_tensor(size_t i) -> Tensor<U>*;

  [[nodiscard]] auto needs_input_grad(size_t i) const -> bool;

  [[nodiscard]] auto num_inputs() const -> size_t;

  auto tensor_grad_fn(size_t i) -> GradFn<T>*;

  auto accumulate_output_grad(const Tensor<T>& output_grad) -> void;

  auto accumulate_output_grad(size_t i, const Tensor<T>& output_grad) -> void;

  auto output_grad() -> const Tensor<T>&;

  template <typename... Args>
  auto set_inputs(Args&... args) -> void;

  auto has_grad_fn(size_t i) -> bool;

private:
  template <typename Input>
  auto set_input_helper(Input& input) -> void;

  template <typename Input>
  auto save_for_backward_helper(Input& item);

  struct InputInfo {
    bool needs_grad;
    GradFn<T>* grad_fn;
    std::optional<Tensor<T>> grad;
  };

  std::optional<Tensor<T>> output_grad_;
  std::vector<std::unique_ptr<TensorBase>> saved_tensors_;
  std::vector<InputInfo> inputs_{};
};

template <typename T>
class GradFn {
public:
  virtual ~GradFn() = default;
  virtual auto backward() -> void = 0;
  virtual auto accumulate_output_grad(const Tensor<T>& output_grad) -> void = 0;
  virtual auto build_topo(std::set<GradFn*>&, std::list<GradFn*>&) -> void = 0;
};

template <typename Function, typename T>
class ContextGradFn : public GradFn<T> {
public:
  explicit ContextGradFn(BackwardContext<T> context) : context_(std::move(context)) {}

  auto backward() -> void override {
    auto grads = Function::backward(context_, context_.output_grad());
    constexpr auto num_grads = std::tuple_size_v<decltype(grads)>;
    iterate_tuple(std::move(grads), std::make_index_sequence<num_grads>{});
  }

  auto accumulate_output_grad(const Tensor<T>& output_grad) -> void override {
    context_.accumulate_output_grad(output_grad);
  }

  auto build_topo(std::set<GradFn<T>*>& visited, std::list<GradFn<T>*>& topo) -> void override {
    if (!visited.contains(this)) {
      visited.insert(this);
      for (size_t i = 0; i < context_.num_inputs(); ++i) {
        if (context_.has_grad_fn(i)) {
          context_.tensor_grad_fn(i)->build_topo(visited, topo);
        }
      }
      topo.push_back(this);
    }
  }

protected:
  template <std::size_t I, typename... Grads>
  auto process_grad(std::tuple<Grads...>& grads) -> void {
    if (context_.needs_input_grad(I) && std::get<I>(grads).has_value()) {
      context_.accumulate_output_grad(I, std::move(*std::get<I>(grads)));
    }
  }

  template <size_t... Is, typename... Grads>
  auto iterate_tuple(std::tuple<Grads...>&& grads, std::index_sequence<Is...>) -> void {
    ((process_grad<Is>(grads)), ...);
  }

  BackwardContext<T> context_;
};

template <typename Derived>
class Function {
public:
  virtual ~Function() = default;

  template <typename T, typename... Rest>
  static auto apply(const Tensor<T>& input, const Rest&... rest) {
    BackwardContext<T> context;
    context.set_inputs(input, rest...);
    auto ret = Derived::forward(context, input, rest...);
    ret.grad_fn_ = std::make_unique<ContextGradFn<Derived, T>>(std::move(context));
    return std::move(ret);
  }
};

template <typename T>
class AddBackward final : public ContextGradFn<AddBackward<T>, T> {
  using ContextGradFn = ContextGradFn<AddBackward<T>, T>;
public:
  AddBackward(const Tensor<T>& a, const Tensor<T>& b) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(a, b);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    std::optional<Tensor<T>> input_grad;
    std::optional<Tensor<T>> other_grad;
    if (context.needs_input_grad(0)) {
      input_grad = output_grad.view();
    }
    if (context.needs_input_grad(1)) {
      other_grad = output_grad.view();
    }
    return std::make_tuple(std::move(input_grad), std::move(other_grad));
  }
};

template <typename T>
class SubBackward final : public ContextGradFn<SubBackward<T>, T> {
  using ContextGradFn = ContextGradFn<SubBackward<T>, T>;
public:
  SubBackward(const Tensor<T>& a, const Tensor<T>& b) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(a, b);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    std::optional<Tensor<T>> input_grad;
    std::optional<Tensor<T>> other_grad;
    if (context.needs_input_grad(0)) {
      input_grad = output_grad.view();
    }
    if (context.needs_input_grad(1)) {
      other_grad = -output_grad;
    }
    return std::make_tuple(std::move(input_grad), std::move(other_grad));
  }
};

template <typename T>
class DivBackward final : public ContextGradFn<DivBackward<T>, T> {
  using ContextGradFn = ContextGradFn<DivBackward<T>, T>;
public:
  DivBackward(const Tensor<T>& a, const Tensor<T>& b) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(a, b);
    ContextGradFn::context_.save_for_backward(a);
    ContextGradFn::context_.save_for_backward(b);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& a = *context.template saved_tensor<T>(0);
    auto& b = *context.template saved_tensor<T>(1);
    std::optional<Tensor<T>> input_grad;
    std::optional<Tensor<T>> other_grad;
    if (context.needs_input_grad(0)) {
      // w.r.t. a = 1 / b
      input_grad = static_cast<T>(1) / b * output_grad;
    }
    if (context.needs_input_grad(1)) {
      // w.r.t. b = -a / b^2
      other_grad = -(a / b.pow(2)) *  output_grad;
    }
    return std::make_tuple(std::move(input_grad), std::move(other_grad));
  }
};

template <typename T>
class ExpBackward final : public ContextGradFn<ExpBackward<T>, T> {
  using ContextGradFn = ContextGradFn<ExpBackward<T>, T>;
public:
  ExpBackward(const Tensor<T>& x, const Tensor<T>& out) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(x);
    ContextGradFn::context_.save_for_backward(out);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& out = *context.template saved_tensor<T>(0);
    std::optional<Tensor<T>> input_grad;
    if (context.needs_input_grad(0)) {
      input_grad = out * output_grad;
    }
    return std::make_tuple(std::move(input_grad));
  }
};

template <typename T>
class TanhBackward final : public ContextGradFn<TanhBackward<T>, T> {
  using ContextGradFn = ContextGradFn<TanhBackward<T>, T>;
public:
  TanhBackward(const Tensor<T>& x, const Tensor<T>& out) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(x);
    ContextGradFn::context_.save_for_backward(out);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& out = *context.template saved_tensor<T>(0);
    std::optional<Tensor<T>> input_grad;
    if (context.needs_input_grad(0)) {
      input_grad = (static_cast<T>(1) - out.pow(static_cast<T>(2))) * output_grad;
    }
    return std::make_tuple(std::move(input_grad));
  }
};

template <typename T>
class LogBackward final : public ContextGradFn<LogBackward<T>, T> {
  using ContextGradFn = ContextGradFn<LogBackward<T>, T>;
public:
  explicit LogBackward(const Tensor<T>& x) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(x);
    ContextGradFn::context_.save_for_backward(x);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& x = *context.template saved_tensor<T>(0);
    std::optional<Tensor<T>> input_grad;
    if (context.needs_input_grad(0)) {
      input_grad = output_grad / x;
    }
    return std::make_tuple(std::move(input_grad));
  }
};

template <typename T>
class MeanBackward final : public ContextGradFn<MeanBackward<T>, T> {
  using ContextGradFn = ContextGradFn<MeanBackward<T>, T>;
public:
  explicit MeanBackward(const Tensor<T>& x) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(x);
    ContextGradFn::context_.save_for_backward(x);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& x = *context.template saved_tensor<T>(0);
    std::optional<Tensor<T>> input_grad;
    if (context.needs_input_grad(0)) {
      Tensor<T> delta(x.context(), x.dims());
      delta.fill_(output_grad.item() / static_cast<T>(x.data_size()));
      input_grad = std::move(delta);
    }
    return std::make_tuple(std::move(input_grad));
  }
};

template <typename T>
class MulBackward final : public ContextGradFn<MulBackward<T>, T> {
  using ContextGradFn = ContextGradFn<MulBackward<T>, T>;
public:
  MulBackward(const Tensor<T>& input, const Tensor<T>& other) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(input, other);
    ContextGradFn::context_.save_for_backward(input);
    ContextGradFn::context_.save_for_backward(other);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& input = *context.template saved_tensor<T>(0);
    auto& other = *context.template saved_tensor<T>(1);
    std::optional<Tensor<T>> input_grad;
    std::optional<Tensor<T>> other_grad;
    if (context.needs_input_grad(0)) {
      input_grad = other * output_grad;
    }
    if (context.needs_input_grad(1)) {
      other_grad = input * output_grad;
    }
    return std::make_tuple(std::move(input_grad), std::move(other_grad));
  }
};

template <typename T>
class NegBackward final : public ContextGradFn<NegBackward<T>, T> {
  using ContextGradFn = ContextGradFn<NegBackward<T>, T>;
public:
  explicit NegBackward(const Tensor<T>& x) : ContextGradFn(BackwardContext<T>{}) {
    ContextGradFn::context_.set_inputs(x);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    std::optional<Tensor<T>> input_grad;
    if (context.needs_input_grad(0)) {
      input_grad = -output_grad;
    }
    return std::make_tuple(std::move(input_grad));
  }
};

template <typename T>
class PowBackward final : public ContextGradFn<PowBackward<T>, T> {
  using ContextGradFn = ContextGradFn<PowBackward<T>, T>;
public:
  explicit PowBackward(const Tensor<T>& x, T scalar) : ContextGradFn(BackwardContext<T>{}) {
    Tensor y(x.context(), scalar);
    ContextGradFn::context_.set_inputs(x, y);
    ContextGradFn::context_.save_for_backward(x, y);
  }

  static auto backward(BackwardContext<T>& context, const Tensor<T>& output_grad) {
    auto& input = *context.template saved_tensor<T>(0);
    auto& scalar = *context.template saved_tensor<T>(1);
    std::optional<Tensor<T>> input_grad;
    if (context.needs_input_grad(0)) {
      input_grad = scalar * input.pow(scalar.item() - 1) * output_grad;
    }
    return std::make_tuple(std::move(input_grad));
  }
};

} // namespace damml

#endif  // TENSOR_FUNCTION_H
