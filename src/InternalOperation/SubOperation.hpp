#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class SubOperation : public IOperation<T> {
    public:
        // Forward pass: Element-wise subtraction
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) override {
            // Ensure that there are exactly two columns (two children) for the subtraction
            if (children_values.shape()[0] != 2) {
                throw std::invalid_argument("SubOperation requires exactly two child tensors.");
            }

            // Perform element-wise subtraction of the two columns
            return xt::view(children_values, 0, xt::all()) - xt::view(children_values, 1, xt::all());
        }

        // Backward pass: Gradients of the subtraction
        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) override {
            // Ensure that there are exactly two columns (two children) for the subtraction
            if (children_values.shape()[0] != 2) {
                throw std::invalid_argument("SubOperation requires exactly two child tensors.");
            }

            // Create an array to store the gradients of the children
            xt::xarray<T> children_grads = xt::zeros_like(children_values);

            // Gradient for the first child is the same as the parent's gradient
            xt::view(children_grads, 0, xt::all()) = parent_grads;

            // Gradient for the second child is the negative of the parent's gradient
            xt::view(children_grads, 1, xt::all()) = -parent_grads;

            return children_grads;
        }
    };
} // namespace nn::Operation
