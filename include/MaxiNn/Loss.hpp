#pragma once
#include "../MaxiNn.hpp"

namespace nn::loss
{
    template <typename T>
    std::shared_ptr<nn::tensor::Tensor<T>> meanSquaredError(std::shared_ptr<nn::tensor::Tensor<T>> pred, std::shared_ptr<nn::tensor::Tensor<T>> real) {
        auto sub = nn::math::pow(pred - real, static_cast<T>(2));
        return nn::math::reduceMean(sub);
    }

    template <typename T>
    std::shared_ptr<nn::tensor::Tensor<T>> meanAbsoluteError(std::shared_ptr<nn::tensor::Tensor<T>> pred, std::shared_ptr<nn::tensor::Tensor<T>> real) {
        auto sub = nn::math::abs(pred - real);
        return nn::math::reduceMean(sub);
    }

    template <typename T>
    std::shared_ptr<nn::tensor::Tensor<T>> categoricalCrossEntropy(std::shared_ptr<nn::tensor::Tensor<T>> pred, std::shared_ptr<nn::tensor::Tensor<T>> real, T epsilon = 1e-10) {
        // to avoid log(0)
        auto log_prod = nn::math::log(pred + epsilon) * real;

        // sum over the axis 1 to have (bs, )
        T opp = (T)-1;
        auto b_cce = opp * nn::math::reduceSum(log_prod, {1});

        return nn::math::reduceMean(b_cce);
    }
} // namespace nn::Loss
