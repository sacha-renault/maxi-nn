#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class AddOperation : public IOperation<T> {
    public:
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) override {
            // element wise sum
            xt::xarray<T> result = xt::sum(children_values);
            return result;
        }

        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) override {
            // Create a matrix to store the gradients of the children
            xt::xarray<T> children_grads = xt::zeros_like(children_values);
            
            // Propagate the parent's gradient to each child
            for (int i = 0; i < children_values.shape(0); ++i) {
                xt::view(children_grads, i, xt::all()) = parent_grads;  // For add, the gradient is simply passed through
            }

            return children_grads;
        }
    };
} // namespace nn::Operation
