
#ifndef SRC_UTIL_UTIL_H
#define SRC_UTIL_UTIL_H

#include <ranges>
#include <functional>

namespace damml {

template <typename T>
inline auto view_to_string = std::views::transform([](const T& i) { return std::to_string(i); });

class join_with {
public:
  explicit join_with(std::string delimiter) : delimiter_(std::move(delimiter)) {}

  template <std::ranges::range Range>
  std::string operator()(const Range& input) const {
    std::ostringstream oss;
    bool first = true;
    for (const auto& str : input) {
      if (!first) {
        oss << delimiter_;
      }
      oss << str;
      first = false;
    }
    return oss.str();
  }

  template <std::ranges::range Range>
  friend auto operator|(Range&& r, const join_with& jw) {
    return jw(std::forward<Range>(r));
  }

private:
  std::string delimiter_;
};

template <typename>
struct function_argument_count;

template <typename Ret, typename... Args>
struct function_argument_count<Ret(Args...)> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename Ret, typename... Args>
struct function_argument_count<Ret(*)(Args...)> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename Ret, typename... Args>
struct function_argument_count<std::function<Ret(Args...)>> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename Callable>
struct function_argument_count : function_argument_count<decltype(&Callable::operator())> {};

template <typename ClassType, typename Ret, typename... Args>
struct function_argument_count<Ret(ClassType::*)(Args...) const> {
  static constexpr size_t value = sizeof...(Args);
};

template <typename T>
inline constexpr auto function_argument_count_v = function_argument_count<T>::value;

}  // namespace damml

#endif  // SRC_UTIL_UTIL_H
