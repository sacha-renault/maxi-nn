#pragma once
#include <functional>
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class ElementWiseOperation : public IOperation<T> {
    private:
        std::function<xt::xarray<T>(const xt::xarray<T>&)> forward_func;
        std::function<xt::xarray<T>(const xt::xarray<T>&)> backward_func;

    public:
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) override {
            if (children_values.shape()[0] != 1) {
                throw std::invalid_argument("Element-wise operation must have exactly one child tensor.");
            }

            // Extract the single child tensor
            xt::xarray<T> single_child = xt::view(children_values, 0, xt::all());
            return forward_func(single_child);
        }

        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) override {
            if (children_values.shape()[0] != 1) {
                throw std::invalid_argument("Element-wise operation must have exactly one child tensor.");
            }

            // Extract the single child tensor
            // xt::xarray<T> single_child = xt::view(children_values, 0, xt::all());

            // Compute local gradients using the backward function
            xt::xarray<T> local_grads = backward_func(children_values);

            // Compute the gradients for the child
            xt::xarray<T> children_grads = local_grads * parent_grads;

            return children_grads;
        }

        // Constructor to initialize the forward and backward functions
        ElementWiseOperation(
            std::function<xt::xarray<T>(const xt::xarray<T>&)> f,
            std::function<xt::xarray<T>(const xt::xarray<T>&)> b
        ) : forward_func(f), backward_func(b) { }
    };
} // namespace nn::Operation
