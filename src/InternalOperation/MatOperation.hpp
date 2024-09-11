#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    std::shared_ptr<IOperation<T>> Dot = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Dot must have exactly two children");
            }
            const auto& lhs = children_values[0].get();
            const auto& rhs = children_values[1].get();
            // Forward pass: Compute the dot product (matrix multiplication)
            return xt::linalg::dot(lhs, rhs);
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Dot must have exactly two children");
            }
            const auto& lhs = children_values[0].get();
            const auto& rhs = children_values[1].get();

            // Backward pass: Compute gradients for each input
            // Gradients are computed using the chain rule for matrix multiplication
            auto grad_wrt_lhs = xt::linalg::dot(output_grads, xt::transpose(rhs));
            auto grad_wrt_rhs = xt::linalg::dot(xt::transpose(lhs), output_grads);

            return {grad_wrt_lhs, grad_wrt_rhs};
        }
    );
} // namespace nn::Operation
