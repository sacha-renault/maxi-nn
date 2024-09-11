#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    std::shared_ptr<IOperation<T>> Tanh = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Tanh must have exactly one child");
            }
            // Forward function: Compute hyperbolic tangent element-wise
            return xt::tanh(children_values[0].get());
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Tanh must have exactly one child");
            }
            // Backward function: Compute gradient w.r.t the input
            // Gradient of Tanh(x) is (1 - Tanh(x)^2)
            auto tanh_val = output_vals;
            auto grad_wrt_input = output_grads * (1 - xt::square(tanh_val));
            
            return {grad_wrt_input};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> ReLU = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 1) {
                throw std::runtime_error("ReLU must have exactly one child");
            }
            // Forward pass: Apply ReLU function
            return xt::maximum(children_values[0].get(), static_cast<T>(0));
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 1) {
                throw std::runtime_error("ReLU must have exactly one child");
            }
            // Backward pass: Apply gradient where input was positive
            auto input_vals = children_values[0].get();
            auto grad_wrt_input = output_grads * (input_vals > static_cast<T>(0));
            return {grad_wrt_input};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Pow(T power) {
        return std::make_shared<IOperation<T>>(
            [power](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("Pow must have exactly one child");
                }
                // Forward pass: Apply power function
                return xt::pow(children_values[0].get(), power);
            },
            [power](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
                    const xt::xarray<T>& output_vals,
                    const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("Pow must have exactly one child");
                }
                // Backward pass: Apply gradient calculation for power function
                auto input_vals = children_values[0].get();
                auto grad_wrt_input = output_grads * (power * xt::pow(input_vals, power - 1));
                return {grad_wrt_input};
            }
        );
    }

    template <typename T>
    std::shared_ptr<IOperation<T>> Abs = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Abs must have exactly one child");
            }
            // Forward pass: Apply absolute value function
            return xt::abs(children_values[0].get());
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Abs must have exactly one child");
            }
            // Backward pass: Gradient calculation for absolute value function
            auto input_vals = children_values[0].get();
            auto grad_wrt_input = output_grads * xt::sign(input_vals);
            return {grad_wrt_input};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Exp = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Exp must have exactly one child");
            }
            // Forward pass: Apply exponential function
            return xt::exp(children_values[0].get());
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Exp must have exactly one child");
            }
            // Backward pass: Gradient calculation for exponential function
            auto grad_wrt_input = output_grads * output_vals; // d(exp(x))/dx = exp(x)
            return {grad_wrt_input};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Log = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Log must have exactly one child");
            }
            // Forward pass: Apply natural logarithm function
            return xt::log(children_values[0].get());
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Log must have exactly one child");
            }
            // Backward pass: Gradient calculation for logarithm function
            auto grad_wrt_input = output_grads / children_values[0].get(); // d(log(x))/dx = 1/x
            return {grad_wrt_input};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Sqrt = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Sqrt must have exactly one child");
            }
            // Forward pass: Apply square root function
            return xt::sqrt(children_values[0].get());
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 1) {
                throw std::runtime_error("Sqrt must have exactly one child");
            }
            // Backward pass: Gradient calculation for square root function
            auto input = children_values[0].get();
            auto grad_wrt_input = output_grads / (2 * xt::sqrt(input)); // d(sqrt(x))/dx = 1/(2*sqrt(x))
            return {grad_wrt_input};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Clip(T min, T max) {
        return std::make_shared<IOperation<T>>(
            [min, max](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("Clip must have exactly one child");
                }
                const auto& input = children_values[0].get();
                // Forward pass: Clip the values to be within the range [min, max]
                return xt::clip(input, min, max);
            },
            [min, max](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
                       const xt::xarray<T>& output_vals,
                       const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
                if (children_values.size() != 1) {
                    throw std::runtime_error("Clip must have exactly one child");
                }
                const auto& input = children_values[0].get();
                // Backward pass: Gradient is 0 where input is clipped, 1 elsewhere
                auto grad_wrt_input = xt::where(
                    (input < min) | (input > max),
                    xt::xarray<T>(0),
                    output_grads
                );
                return {grad_wrt_input};
            }
        );
    }

    template <typename T>
    std::shared_ptr<IOperation<T>> reduceMean(xt::dynamic_shape<size_t> axis = xt::dynamic_shape<size_t>()) {
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
                    // Expand gradients to the shape of the original input
                    grad_wrt_input[0] = xt::broadcast(output_grads, child.shape()) / xt::prod(child.shape(axis));
                }
                return grad_wrt_input;
            }
        );
    }
} // namespace nn::Operation
