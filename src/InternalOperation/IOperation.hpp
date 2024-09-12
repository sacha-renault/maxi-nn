#pragma once
#include <functional>
#include <vector>
#include <xtensor/xarray.hpp>

namespace nn::Operation
{
    template <typename T>
    class IOperation
    {
    private:
        std::function<xt::xarray<T>(const std::vector<std::reference_wrapper<const xt::xarray<T>>>&)> forward_func;
        std::function<std::vector<xt::xarray<T>>(const std::vector<std::reference_wrapper<const xt::xarray<T>>>&, const xt::xarray<T>&, const xt::xarray<T>&)> backward_func;

    public:
        virtual ~IOperation() = default;

        IOperation(
            std::function<xt::xarray<T>(const std::vector<std::reference_wrapper<const xt::xarray<T>>>&)> forward,
            std::function<std::vector<xt::xarray<T>>(const std::vector<std::reference_wrapper<const xt::xarray<T>>>&, const xt::xarray<T>&, const xt::xarray<T>&)> backward)
            : forward_func(std::move(forward)), backward_func(std::move(backward)) {}

        xt::xarray<T> forward(const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values) {
            return forward_func(children_values);
        }

        std::vector<xt::xarray<T>> backward(
            const std::vector<std::reference_wrapper<const xt::xarray<T>>>& children_values,
            const xt::xarray<T>& parent_values,
            const xt::xarray<T>& parent_grads) {
            return backward_func(children_values, parent_values, parent_grads);
        }
    };
} // namespace nn::Operation
