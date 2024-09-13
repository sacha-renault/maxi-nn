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

    template <typename T>
    std::shared_ptr<Tensor<T>> categoricalCrossEntropy(std::shared_ptr<Tensor<T>> pred, std::shared_ptr<Tensor<T>> real, T epsilon = 1e-10) {
        // to avoid log(0)
        auto log_prod = math::log(pred + epsilon) * real;

        // sum over the axis 1 to have (bs, )
        T opp = (T)-1;
        auto b_cce = opp * math::reduceSum(log_prod, {1});

        return math::reduceMean(b_cce);
    }
} // namespace nn::Loss
