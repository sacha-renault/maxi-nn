#pragma once
#include <functional>
#include <vector>

namespace nn::Operation
{
    template <typename T>
    class IOperation {
    public:
        virtual ~IOperation() = default;

        /// @brief Compute the values for output tensor from children values
        /// @param children_values
        /// @return values to set for output node
        virtual xt::xarray<T> forward(xt::xarray<T>& children_values) = 0;

        /// @brief Compute the gradient to accumulate for children
        /// @param children_values values of children
        /// @param parent_values values of parent
        /// @param parent_grads grad of parent
        /// @return a matrix containing gradient to accumulate on children nodes
        virtual xt::xarray<T> backward(
            xt::xarray<T>& children_values,
            xt::xarray<T>& parent_values,
            xt::xarray<T>& parent_grads
        ) = 0;
    };
} // namespace nn::Operation
