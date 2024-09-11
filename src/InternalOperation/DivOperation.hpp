#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class DivOperation : public IOperation<T> {
    public:
        // Forward pass: Element-wise division
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) override {
            // Ensure there are exactly two child tensors
            if (children_values.shape()[0] != 2) {
                throw std::invalid_argument("DivOperation requires exactly two child tensors.");
            }
            // Perform element-wise division
            return xt::view(children_values, 0, xt::all()) / xt::view(children_values, 1, xt::all());
        }

        // Backward pass: Gradient of the division
        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) override {
            // Ensure there are exactly two child tensors
            if (children_values.shape()[0] != 2) {
                throw std::invalid_argument("DivOperation requires exactly two child tensors.");
            }

            // Extract numerator and denominator
            xt::xarray<T> numerator = xt::view(children_values, 0, xt::all());
            xt::xarray<T> denominator = xt::view(children_values, 1, xt::all());

            // Create a matrix to store the gradients of the two children
            xt::xarray<T> grads = xt::zeros_like(children_values);

            // Gradient with respect to the numerator (child 0)
            xt::view(grads, 0, xt::all()) = parent_grads / denominator;

            // Gradient with respect to the denominator (child 1)
            xt::view(grads, 1, xt::all()) = -parent_grads * numerator / xt::square(denominator);

            return grads;
        }
    };
} // namespace nn::Operation
