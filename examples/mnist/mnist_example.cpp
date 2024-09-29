#include <iostream>

#include "damml.hpp"
#include "tensor/tensor.hpp"

using damml::Damml;
using damml::Tensor;

auto main() -> int {
    std::cout << "mnist example\n";

    const Damml damml;
    const std::vector<size_t> sz {10, 10};
    const Tensor t = damml.tensor<double>(sz);
    std::cout << "t: " << t << "\n";
    return 0;
}
