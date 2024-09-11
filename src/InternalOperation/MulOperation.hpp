#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class MulOperation : public IOperation<T> {
    public:
        // Forward pass: Element-wise multiplication
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) override {
            // Ensure there are exactly two child tensors
            if (children_values.shape()[0] != 2) {
                throw std::invalid_argument("MulOperation requires exactly two child tensors.");
            }
            // Perform element-wise multiplication
            return xt::view(children_values, 0, xt::all()) * xt::view(children_values, 1, xt::all());
        }

        // Backward pass: Gradient of the multiplication
        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) override {
            // Ensure there are exactly two child tensors
            if (children_values.shape()[0] != 2) {
                throw std::invalid_argument("MulOperation requires exactly two child tensors.");
            }

            // Create an array to store the gradients of the children
            xt::xarray<T> children_grads = xt::zeros_like(children_values);

            // Grad for child 1 is parent_grads * value of child 2
            xt::view(children_grads, 0, xt::all()) = parent_grads * xt::view(children_values, 1, xt::all());

            // Grad for child 2 is parent_grads * value of child 1
            xt::view(children_grads, 1, xt::all()) = parent_grads * xt::view(children_values, 0, xt::all());

            return children_grads;
        }
    };
} // namespace nn::Operation
