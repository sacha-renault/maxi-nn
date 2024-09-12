#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    std::shared_ptr<IOperation<T>> Add2 = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Add2 must have exactly two children");
            }
            return children_values[0].get() + children_values[1].get();
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
        const xt::xarray<T>& output_vals,
        const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Add2 must have exactly two children");
            }
            return {output_grads, output_grads};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Div2 = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Div2 must have exactly two children");
            }
            // Forward function: Element-wise division of two tensors
            return children_values[0].get() / children_values[1].get();
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
           const xt::xarray<T>& output_vals,
           const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Div2 must have exactly two children");
            }
            xt::xarray<T> grad_wrt_first = output_grads / children_values[1].get();
            xt::xarray<T> grad_wrt_second = -output_grads * children_values[0].get() / (children_values[1].get() * children_values[1].get());
            return {grad_wrt_first, grad_wrt_second};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Mul2 = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Mul2 must have exactly two children");
            }
            return children_values[0].get() * children_values[1].get();
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
        const xt::xarray<T>& output_vals,
        const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Mul2 must have exactly two children");
            }
            // Backward function:
            xt::xarray<T> grad_wrt_first = output_grads * children_values[1].get();
            xt::xarray<T> grad_wrt_second = output_grads * children_values[0].get();
            return {grad_wrt_first, grad_wrt_second};
        }
    );

    template <typename T>
    std::shared_ptr<IOperation<T>> Sub2 = std::make_shared<IOperation<T>>(
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) -> xt::xarray<T> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Sub2 must have exactly two children");
            }
            return children_values[0].get() - children_values[1].get();
        },
        [](const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
        const xt::xarray<T>& output_vals,
        const xt::xarray<T>& output_grads) -> std::vector<xt::xarray<T>> {
            if (children_values.size() != 2) {
                throw std::runtime_error("Sub2 must have exactly two children");
            }
            return {output_grads, -output_grads};
        }
    );
} // namespace nn::Operation
