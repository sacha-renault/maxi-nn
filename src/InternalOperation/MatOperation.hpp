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
                    int i = 0;
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
                    int i = 0;
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
    std::shared_ptr<IOperation<T>> Softmax = std::make_shared<IOperation<T>>(
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

            // // Use the Jacobian matrix of the softmax function
            // auto eye = xt::eye(output_vals.shape(1));

            // // Compute the Jacobian
            // auto J = xt::expand_dims(output_vals, 1) * ((xt::expand_dims(eye, 0) - xt::expand_dims(output_vals, 2)));
            // // std::cout << xt::adapt(J.shape()) << std::endl;
            // // std::cout << xt::adapt(J.shape()) << std::endl;
            // auto localGrad = xt::sum(J, {1}); // sum over z axis to get same shape as grad of tensor

            // // computation of global gradient
            // auto d_softmax = localGrad * output_grads;

            // std::cout << "\nlocal : " << std::endl;
            // std::cout << localGrad << std::endl;
            // std::cout << "\nparent grad : " << std::endl;
            // std::cout << output_grads << std::endl;
            // std::cout << "\nglobal : " << std::endl;
            // std::cout << d_softmax << std::endl;

            // return {d_softmax};

            // Get the input logits (before softmax)
            const xt::xarray<T>& logits = children_values[0].get();

            // Get the softmax output (after softmax)
            const xt::xarray<T>& softmax_output = output_vals;

            // Initialize the gradient with respect to input
            xt::xarray<T> input_grads = xt::zeros_like(logits);

            // Iterate through each element to compute the gradient
            for (std::size_t i = 0; i < softmax_output.shape()[0]; ++i) {
                for (std::size_t j = 0; j < softmax_output.shape()[1]; ++j) {
                    for (std::size_t k = 0; k < softmax_output.shape()[1]; ++k) {
                        if (j == k) {
                            input_grads(i, j) += output_grads(i, k) * softmax_output(i, j) * (1 - softmax_output(i, j));
                        } else {
                            input_grads(i, j) -= output_grads(i, k) * softmax_output(i, j) * softmax_output(i, k);
                        }
                    }
                }
            }

            // Return the gradient with respect to the input logits
            return {input_grads};
        }
    );
} // namespace nn::Operation
