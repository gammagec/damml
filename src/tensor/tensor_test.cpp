#define BOOST_TEST_MODULE TensorTest

#include <iostream>
#include <vector>
#include <boost/test/tools/detail/print_helper.hpp>

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

  BOOST_FIXTURE_TEST_CASE(initialize_shape_zeros, TestFixture) {
    const std::vector<size_t> sz{10, 10};
    const Tensor t = damml.tensor<double>(sz);
    for (size_t i = 0; i < 10; ++i) {
      for (size_t j = 0; j < 10; ++j) {
        BOOST_CHECK_EQUAL((t[i, j].item()), 0);
      }
    }
    BOOST_CHECK_EQUAL(t.dims(), (std::vector<size_t>{10, 10}));
  }

  BOOST_FIXTURE_TEST_CASE(initialize_with_values, TestFixture) {
    const auto t = make_1d();
    BOOST_CHECK_EQUAL((t[0].item()), 1.);
    BOOST_CHECK_EQUAL((t[1].item()), 2.);
    BOOST_CHECK_EQUAL((t[2].item()), 3.);
    BOOST_CHECK_EQUAL((t[3].item()), 4.);
    BOOST_CHECK_EQUAL(t.dims(), std::vector<size_t>{4});
  }

  BOOST_FIXTURE_TEST_CASE(oob_1d_access, TestFixture) {
    const auto t = make_1d();
    BOOST_CHECK_THROW((t[4].item()), OobException);
  }

  BOOST_FIXTURE_TEST_CASE(initialize_2d_with_values, TestFixture) {
    const auto t = make_2d();
    BOOST_CHECK_EQUAL((t[0, 0].item()), 1.);
    BOOST_CHECK_EQUAL((t[1, 1].item()), 2.1);
    BOOST_CHECK_EQUAL((t[2, 2].item()), 3.2);
    BOOST_CHECK_EQUAL((t[3, 3].item()), 4.3);
    BOOST_CHECK_EQUAL(t.dims(), (std::vector<size_t>{4, 4}));
  }

  BOOST_FIXTURE_TEST_CASE(oob_2d_access, TestFixture) {
    const auto t = make_2d();
    BOOST_CHECK_THROW((t[3, 4].item()), OobException);
  }

  BOOST_FIXTURE_TEST_CASE(equals_tensor, TestFixture) {
    const auto t1 = make_2d();
    const auto t2 = make_2d();
    BOOST_CHECK(t1 == t2);
  }

  BOOST_FIXTURE_TEST_CASE(equals_tensor_false, TestFixture) {
    const auto t1 = make_2d();
    const auto t2 = damml.tensor({
      {1.0, 2.0, 3.0, 4.0},
      {1.1, 2.1, 3.1, 4.1},
      {1.2, 2.2, 3.2, 4.2},
      {1.3, 2.3, 3.3, 4.4},
    });
    BOOST_CHECK(t1 != t2);
  }

  BOOST_FIXTURE_TEST_CASE(matmul, TestFixture) {
    const auto t1 = damml.tensor({
      {1, 0, 1},
      {2, 1, 1},
      {0, 1, 1},
      {1, 1, 2}
    });
    const auto t2 = damml.tensor({
      {1, 2, 1},
      {2, 3, 1},
      {4, 2, 2},
    });
    BOOST_CHECK_EQUAL(t1.mm(t2), damml.tensor({
                        {5, 4, 3},
                        {8, 9, 5},
                        {6, 5, 3},
                        {11, 9, 6}
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(matmul_1d, TestFixture) {
    const auto t1 = damml.tensor({
      {1, 0, 1},
      {2, 1, 1},
      {0, 1, 1},
      {1, 1, 2}
    });
    const auto t2 = damml.tensor({1, 2, 1});
    BOOST_CHECK_EQUAL(t1.matmul(t2), damml.tensor({2, 5, 3, 5}));
  }

  BOOST_FIXTURE_TEST_CASE(transpose, TestFixture) {
    auto t = make_2d();
    BOOST_CHECK_EQUAL(t.t(), damml.tensor({
                        {1.0, 1.1, 1.2, 1.3},
                        {2.0, 2.1, 2.2, 2.3},
                        {3.0, 3.1, 3.2, 3.3},
                        {4.0, 4.1, 4.2, 4.3}
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(add_inplace, TestFixture) {
    auto t = make_2d();
    t.add_(damml.tensor({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    }));
    BOOST_CHECK_EQUAL(t, damml.tensor({
                        {2.0, 3.0, 4.0, 5.0},
                        {3.1, 4.1, 5.1, 6.1},
                        {4.2, 5.2, 6.2, 7.2},
                        {5.3, 6.3, 7.3, 8.3}
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(neg, TestFixture) {
    const auto t = make_2d();
    BOOST_CHECK_EQUAL(t.neg(), damml.tensor({
                        {-1.0, -2.0, -3.0, -4.0},
                        {-1.1, -2.1, -3.1, -4.1},
                        {-1.2, -2.2, -3.2, -4.2},
                        {-1.3, -2.3, -3.3, -4.3},
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(exp, TestFixture) {
    const auto t = damml.tensor<double>({0, 0});
    BOOST_CHECK_EQUAL(t.exp(), damml.tensor<double>({1, 1}));
  }

  BOOST_FIXTURE_TEST_CASE(add, TestFixture) {
    auto t = make_2d();
    BOOST_CHECK_EQUAL(t.add(damml.tensor({
                        {1., 1., 1., 1.},
                        {2., 2., 2., 2.},
                        {3., 3., 3., 3.},
                        {4., 4., 4., 4.}
                        })), damml.tensor({
                        {2.0, 3.0, 4.0, 5.0},
                        {3.1, 4.1, 5.1, 6.1},
                        {4.2, 5.2, 6.2, 7.2},
                        {5.3, 6.3, 7.3, 8.3}
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(mul, TestFixture) {
    auto t = make_2d();
    BOOST_TENSOR_EQUALS_APPROX(t.mul(
                                 damml.tensor({
                                   {1., 1., 1., 1.},
                                   {2., 2., 2., 2.},
                                   {3., 3., 3., 3.},
                                   {4., 4., 4., 4.}
                                   })),
                               damml.tensor({
                                 {1.0, 2.0, 3.0, 4.0},
                                 {2.2, 4.2, 6.2, 8.2},
                                 {3.6, 6.6, 9.6, 12.6},
                                 {5.2, 9.2, 13.2, 17.2}
                                 }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(div, TestFixture) {
    auto t = make_2d();
    BOOST_TENSOR_EQUALS_APPROX(t.div(damml.tensor({
                                 {2., 2., 2., 2.},
                                 {1., 1., 1., 1.},
                                 {2., 2., 2., 2.},
                                 {1., 1., 1., 1.},
                                 })), damml.tensor({
                                 {.5, 1., 1.5, 2.},
                                 {1.1, 2.1, 3.1, 4.1},
                                 {.6, 1.1, 1.6, 2.1},
                                 {1.3, 2.3, 3.3, 4.3}
                                 }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(unsqueeze, TestFixture) {
    const auto t = damml.tensor({1, 2, 3, 4, 5});
    BOOST_CHECK_EQUAL(t.unsqueeze(0).dims(), (std::vector<size_t>{1, 5}));
  }

  BOOST_FIXTURE_TEST_CASE(unsqueeze2, TestFixture) {
    const auto t = damml.tensor({1, 2, 3, 4, 5});
    BOOST_CHECK_EQUAL(t.unsqueeze(1).dims(), (std::vector<size_t>{5, 1}));
  }

  BOOST_FIXTURE_TEST_CASE(expand, TestFixture) {
    const auto t = damml.tensor({
      {1},
      {2},
      {3},
    });
    BOOST_CHECK_EQUAL(t.expand({3, 4}), damml.tensor({
                        {1, 1, 1, 1},
                        {2, 2, 2, 2},
                        {3, 3, 3, 3},
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(expand2, TestFixture) {
    const auto t = damml.tensor({{1, 2, 3}});
    BOOST_CHECK_EQUAL(t.expand({2, 3}), damml.tensor({
                        {1, 2, 3},
                        {1, 2, 3},
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(expand_neg, TestFixture) {
    const auto t = damml.tensor({{1, 2, 3}});
    BOOST_CHECK_EQUAL(t.expand({2, -1}), damml.tensor({
                        {1, 2, 3},
                        {1, 2, 3},
                        }));
  }

  BOOST_FIXTURE_TEST_CASE(sum_row, TestFixture) {
    const auto t = damml.tensor({
      {1., -1., 2., 3., 4.},
      {2., -1., 2., 7., 2.},
      {2., -1., 3., 3., 4.}
    });
    BOOST_CHECK_EQUAL(t.sum(0, /*keep_dim=*/ false), damml.tensor({5., -3., 7., 13., 10.}));
  }

  BOOST_FIXTURE_TEST_CASE(sum_col, TestFixture) {
    const auto t = damml.tensor({
      {1., -1., 2., 3., 4.},
      {2., -1., 2., 7., 2.},
      {2., -1., 3., 3., 4.}
    });
    BOOST_CHECK_EQUAL(t.sum(1, /*keep_dim=*/ false), damml.tensor({9., 12., 11.}));
  }

  BOOST_FIXTURE_TEST_CASE(max_row, TestFixture) {
    const auto t = damml.tensor({
      {1., -1., 2., 3., 4.},
      {2., -1., 2., 7., 2.},
      {2., -1., 3., 3., 4.}
    });
    const auto [values, indexes] = t.max(0, /*keep_dim=*/ false);
    BOOST_CHECK_EQUAL(values, damml.tensor({2., -1., 3., 7., 4.}));
    BOOST_CHECK_EQUAL(indexes, damml.tensor<size_t>({1, 0, 2, 1, 0}));
  }

  BOOST_FIXTURE_TEST_CASE(max_col, TestFixture) {
    const auto t = damml.tensor({
      {1., -1., 2., 3., 4.},
      {2., -1., 2., 7., 2.},
      {2., -1., 3., 3., 4.}
    });
    const auto [values, indexes] = t.max(1, /*keep_dim=*/ false);
    BOOST_CHECK_EQUAL(values, damml.tensor({4., 7., 4.}));
    BOOST_CHECK_EQUAL(indexes, damml.tensor<size_t>({4, 3, 4}));
  }

  BOOST_FIXTURE_TEST_CASE(add_backward, TestFixture) {
    const auto t1 = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto t2 = damml.tensor({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto t3 = t1.add(t2);
    BOOST_CHECK_EQUAL(t3, damml.tensor({
      {2., 2., 2., 2.},
      {4., 4., 4., 4.},
      {6., 6., 6., 6.},
      {8., 8., 8., 8.}
    }));
    t3.backward(damml.ones<double>({4, 4}));
    BOOST_CHECK_EQUAL(t1.grad(), damml.ones<double>({4, 4}));
  }

  BOOST_FIXTURE_TEST_CASE(sub_backward, TestFixture) {
    const auto t1 = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {5., 6., 7., 8.},
      {4., 4., 4., 4.}
    });
    const auto t2 = damml.tensor({
      {1., 1., 0., 1.},
      {1., 1., 0., 1.},
      {3., 3., 0., 3.},
      {4., 4., 0., 4.}
    });
    const auto t3 = t1.sub(t2);
    BOOST_CHECK_EQUAL(t3, damml.tensor({
      {0., 0., 1., 0.},
      {1., 1., 2., 1.},
      {2., 3., 7., 5.},
      {0., 0., 4., 0.}
    }));
    t3.backward(damml.ones<double>({4, 4}));
    BOOST_CHECK_EQUAL(t1.grad(), damml.ones<double>({4, 4}));
  }

  BOOST_FIXTURE_TEST_CASE(div_backward, TestFixture) {
    const auto t1 = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto t2 = damml.tensor({
      {2., 2., 2., 2.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {2., 2., 2., 2.}
    });
    const auto t3 = t1.div(t2);
    BOOST_CHECK_EQUAL(t3, damml.tensor({
      {.5, .5, .5, .5},
      {1., 1., 1., 1.},
      {1., 1., 1., 1.},
      {2., 2., 2., 2.}
    }));
    t3.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t1.grad(), damml.tensor({
      {.5, .5, .5, .5},
      {.5, .5, .5, .5},
      {.3333, .3333, .3333, .3333},
      {.5, .5, .5, .5}
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(exp_backward, TestFixture) {
    const auto t = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto out = t.exp();
    BOOST_TENSOR_EQUALS_APPROX(out, damml.tensor({
      {2.7183, 2.7183, 2.7183, 2.7183},
      {7.3891, 7.3891, 7.3891, 7.3891},
      {20.0855, 20.0855, 20.0855, 20.0855},
      {54.5981, 54.5981, 54.5981, 54.5981}
    }), EPSILON);
    out.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t.grad(), damml.tensor({
      {2.7183, 2.7183, 2.7183, 2.7183},
      {7.3891, 7.3891, 7.3891, 7.3891},
      {20.0855, 20.0855, 20.0855, 20.0855},
      {54.5981, 54.5981, 54.5981, 54.5981}
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(tanh_backward, TestFixture) {
    const auto t = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto out = t.tanh();
    BOOST_TENSOR_EQUALS_APPROX(out, damml.tensor({
      {.7616, .7616, .7616, .7616},
      {.9640, .9640, .9640, .9640},
      {.9951, .9951, .9951, .9951},
      {.9993, .9993, .9993, .9993}
    }), EPSILON);
    out.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t.grad(), damml.tensor({
      {.4200, .42, .42, .42},
      {.0707, .0707, .0707, .0707},
      {.0099, .0099, .0099, .0099},
      {.0013, .0013, .0013, .0013}
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(log_backward, TestFixture) {
    const auto t = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto out = t.log();
    BOOST_TENSOR_EQUALS_APPROX(out, damml.tensor({
      {0., 0., 0., 0.},
      {.6931, .6931, .6931, .6931},
      {1.0986, 1.0986, 1.0986, 1.0986},
      {1.3863, 1.3863, 1.3863, 1.3863}
    }), EPSILON);
    out.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t.grad(), damml.tensor({
      {1., 1., 1., 1.},
      {.5, .5, .5, .5},
      {.3333, .3333, .3333, .3333},
      {.25, .25, .25, .25}
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(mean_backward, TestFixture) {
    const auto t = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto out = t.mean();
    BOOST_TENSOR_EQUALS_APPROX(out, damml.tensor(2.5), EPSILON);
    out.backward();
    BOOST_TENSOR_EQUALS_APPROX(t.grad(), damml.tensor({
      {.0625, .0625, .0625, .0625},
      {.0625, .0625, .0625, .0625},
      {.0625, .0625, .0625, .0625},
      {.0625, .0625, .0625, .0625},
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(mul_backward, TestFixture) {
    const auto t1 = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto t2 = damml.tensor({
      {2., 2., 2., 2.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {2., 2., 2., 2.}
    });
    const auto t3 = t1.mul(t2);
    BOOST_CHECK_EQUAL(t3, damml.tensor({
      {2., 2., 2., 2.},
      {4., 4., 4., 4.},
      {9., 9., 9., 9.},
      {8., 8., 8., 8.}
    }));
    t3.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t1.grad(), damml.tensor({
      {2., 2., 2., 2.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {2., 2., 2., 2.},
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(neg_backward, TestFixture) {
    const auto t = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto out = t.neg();
    BOOST_TENSOR_EQUALS_APPROX(out, damml.tensor({
      {-1., -1., -1., -1.},
      {-2., -2., -2., -2.},
      {-3., -3., -3., -3.},
      {-4., -4., -4., -4.}
    }), EPSILON);
    out.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t.grad(), damml.tensor({
      {-1., -1., -1., -1.},
      {-1., -1., -1., -1.},
      {-1., -1., -1., -1.},
      {-1., -1., -1., -1.},
    }), EPSILON);
  }

  BOOST_FIXTURE_TEST_CASE(pow_backward, TestFixture) {
    const auto t = damml.tensor_requires_grad({
      {1., 1., 1., 1.},
      {2., 2., 2., 2.},
      {3., 3., 3., 3.},
      {4., 4., 4., 4.}
    });
    const auto out = t.pow(2.0);
    BOOST_CHECK_EQUAL(out, damml.tensor({
      {1., 1., 1., 1.},
      {4., 4., 4., 4.},
      {9., 9., 9., 9.},
      {16., 16., 16., 16.}
    }));
    out.backward(damml.ones<double>({4, 4}));
    BOOST_TENSOR_EQUALS_APPROX(t.grad(), damml.tensor({
      {2., 2., 2., 2.},
      {4., 4., 4., 4.},
      {6., 6., 6., 6.},
      {8., 8., 8., 8.},
    }), EPSILON);
  }
} // namespace
} // namespace damml
