#pragma once
#include <functional>
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class BatchWiseOperation : public IOperation<T> {
    private:
        // Forward function for the entire tensor or specific dimensions
        std::function<xt::xarray<T>(const xt::xarray<T>&)> forward_func;

        // Backward function for the entire tensor or specific dimensions
        std::function<xt::xarray<T>(const xt::xarray<T>&, const xt::xarray<T>&, const xt::xarray<T>&)> backward_func;

    public:
        // Forward pass
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) override {
            return forward_func(children_values);
        }

        // Backward pass
        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) override {
            return backward_func(children_values, parent_values, parent_grads);
        }

        // Constructor to set the forward and backward functions
        BatchWiseOperation(
            std::function<xt::xarray<T>(const xt::xarray<T>&)> f,
            std::function<xt::xarray<T>(const xt::xarray<T>&, const xt::xarray<T>&, const xt::xarray<T>&)> b
        ) : forward_func(f), backward_func(b) { }
    };
} // namespace nn::Operation
