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
                    
                    // Set 1 in the axes where reduction happened
                    auto target = child.shape();
                    for (std::size_t i = 0; i < target.size(); ++i) {
                        if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
                            target[i] = 1;  // Reduction axis will be 1
                        }
                    }

                    xt::xarray<float> output_grads_xarray = output_grads;
                    auto reshaped_grads = output_grads_xarray.reshape(target);
                    auto broadcasted = xt::broadcast(reshaped_grads, child.shape());
                    grad_wrt_input[0] = broadcasted / product;
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
                    // Set 1 in the axes where reduction happened
                    auto target = child.shape();
                    for (std::size_t i = 0; i < target.size(); ++i) {
                        if (std::find(axis.begin(), axis.end(), i) != axis.end()) {
                            target[i] = 1;  // Reduction axis will be 1
                        }
                    }

                    xt::xarray<float> output_grads_xarray = output_grads;
                    auto reshaped_grads = output_grads_xarray.reshape(target);
                    auto broadcasted = xt::broadcast(reshaped_grads, child.shape());
                    grad_wrt_input[0] = broadcasted;
                }
                return grad_wrt_input;
            }
        );
    }

    template <typename T>
    std::shared_ptr<IOperation<T>> SoftmaxDim1 = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            // Ensure 1 child
            if (children_values.size() != 1) {
                throw std::runtime_error("Softmax must have exactly one children");
            }

            // Get ref
            const auto& child = children_values[0].get();

            // Ensure rank
            if (child.shape().size() != 2 || child.shape({1}) == 1) {
                throw std::runtime_error("Softmax must have (bs, num_input) as input");
            }

            // compute softmax batch-wise
            xt::xarray<T> max_vals = xt::amax(child, {1}, xt::keep_dims);
            xt::xarray<T> exp_vals = xt::exp(child - max_vals);
            xt::xarray<T> sum_exp_vals = xt::sum(exp_vals, {1}, xt::keep_dims);

            // Add a small constant to avoid division by zero
            const T epsilon = 1e-10;
            sum_exp_vals += epsilon;

            return exp_vals / sum_exp_vals;
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            // dot prod on last axis
            xt::xarray<T> dot_product = xt::sum(output_vals * output_grads, {1});
            xt::xarray<T> dot_product_expanded = xt::expand_dims(dot_product, 1);

            // Compute the gradient with respect to the input logits
            xt::xarray<T> input_grads = output_grads * output_vals - output_vals * dot_product_expanded;

            // Return the gradient with respect to the input logits
            return {input_grads};
        }
    );
} // namespace nn::Operation
