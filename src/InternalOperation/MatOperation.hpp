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

    template <typename T>
    std::shared_ptr<IOperation<T>> ReduceMean(xt::dynamic_shape<size_t> axis = xt::dynamic_shape<size_t>()) {
        return std::make_shared<IOperation<T>>(
            [axis](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("ReduceMean must have exactly one child");
                }
                const auto& child = children_values[0].get();

                if (axis.empty()) {
                    // If no axis is provided, reduce over all dimensions
                    return xt::mean(child);
                } else {
                    // Reduce along specified axis
                    return xt::mean(child, axis);
                }
            },
            [axis](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
                       const xt::xarray<T>& output_vals,
                       const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("ReduceMean must have exactly one child");
                }
                const auto& child = children_values[0].get();

                // Calculate the gradient with respect to the input tensor
                std::vector<xt::xarray<T>> grad_wrt_input(1);
                if (axis.empty()) {
                    // If no axis is provided, replicate the gradient to match the input shape
                    grad_wrt_input[0] = xt::broadcast(output_grads, child.shape()) / child.size();
                } else {
                    // Compute the product of dimensions along the specified axis
                    auto shape = child.shape();
                    size_t product = 1;
                    for (size_t ax : axis) {
                        product *= shape[ax];
                    }
                    grad_wrt_input[0] = xt::broadcast(output_grads, child.shape()) / product;
                }
                return grad_wrt_input;
            }
        );
    }

    template <typename T>
    std::shared_ptr<IOperation<T>> ReduceSum(xt::dynamic_shape<size_t> axis = xt::dynamic_shape<size_t>()) {
        return std::make_shared<IOperation<T>>(
            [axis](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("ReduceMean must have exactly one child");
                }
                const auto& child = children_values[0].get();

                if (axis.empty()) {
                    // If no axis is provided, reduce over all dimensions
                    return xt::sum(child);
                } else {
                    // Reduce along specified axis
                    return xt::sum(child, axis);
                }
            },
            [axis](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
                       const xt::xarray<T>& output_vals,
                       const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("ReduceMean must have exactly one child");
                }
                const auto& child = children_values[0].get();

                // Calculate the gradient with respect to the input tensor
                std::vector<xt::xarray<T>> grad_wrt_input(1);
                if (axis.empty()) {
                    // If no axis is provided, replicate the gradient to match the input shape
                    grad_wrt_input[0] = xt::broadcast(output_grads, child.shape());
                } else {
                    // Expand gradients to the shape of the original input
                    grad_wrt_input[0] = xt::broadcast(output_grads, child.shape());
                }
                return grad_wrt_input;
            }
        );
    }
} // namespace nn::Operation
