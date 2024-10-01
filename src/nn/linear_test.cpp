#define BOOST_TEST_MODULE LinearTest

#include <iostream>
#include <vector>
#include <boost/test/tools/detail/print_helper.hpp>
#include "nn/linear.h"

#define BOOST_TENSOR_EQUALS_APPROX(t1, t2, tolerance) \
  BOOST_CHECK_MESSAGE((t1).equals_approx((t2), (tolerance)), "Tensors not equal:\n" \
                    << #t1 << " = " << (t1) << "\n" \
                    << #t2 << " = " << (t2)); \

#define EPSILON 0.0001

template <>
struct boost::test_tools::tt_detail::print_log_value<std::vector<size_t>> {
  auto operator()(std::ostream& os, std::vector<size_t> const& value) const -> void {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
      os << value[i];
      if (i != value.size() - 1) {
        os << ", ";
      }
    }
    os << "]";
  }
};

#include <boost/test/unit_test.hpp>

#include "tensor/tensor.hpp"
#include "damml.hpp"

namespace damml {
namespace {

  struct TestFixture {
    Damml damml;

    [[nodiscard]] auto make_1d() const -> Tensor<double> {
      return damml.tensor(std::initializer_list<double>{1., 2., 3., 4.});
    }

    [[nodiscard]] auto make_2d() const -> Tensor<double> {
      return damml.tensor(std::initializer_list<std::initializer_list<double>>{
        {1.0, 2.0, 3.0, 4.0},
        {1.1, 2.1, 3.1, 4.1},
        {1.2, 2.2, 3.2, 4.2},
        {1.3, 2.3, 3.3, 4.3},
      });
    }
  };

  BOOST_FIXTURE_TEST_CASE(initialize_linear, TestFixture) {
    BOOST_CHECK(true);
  }

} // namespace
} // namespace damml