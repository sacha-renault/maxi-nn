#pragma once
#include "../MaxiNn.hpp"

using namespace nn::math;
using namespace nn::tensor;

namespace nn::loss
{
    template <typename T>
    std::shared_ptr<Tensor<T>> meanSquaredError(std::shared_ptr<Tensor<T>> pred, std::shared_ptr<Tensor<T>> real) {
        auto sub = math::pow(pred - real, static_cast<T>(2));
        return math::reduceMean(sub);
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> meanAbsoluteError(std::shared_ptr<Tensor<T>> pred, std::shared_ptr<Tensor<T>> real) {
        auto sub = math::abs(pred - real);
        return math::reduceMean(sub);
    }
} // namespace nn::Loss
