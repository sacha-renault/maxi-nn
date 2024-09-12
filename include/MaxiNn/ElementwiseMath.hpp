#pragma once
#include <memory>
#include "../MaxiNn.hpp"

namespace nn::math
{
    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> relu(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::ReLU<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::ReLU<T>);

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> tanh(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Tanh<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Tanh<T>);

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> pow(std::shared_ptr<tensor::Tensor<T>> input, T exponent) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Pow<T>(exponent)->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Pow<T>(exponent));

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> abs(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Abs<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Abs<T>);

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> exp(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Exp<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Exp<T>);

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> log(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Log<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Log<T>);

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> sqrt(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Sqrt<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Sqrt<T>);

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> clip(std::shared_ptr<tensor::Tensor<T>> input, T min, T max) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Clip<T>(min, max)->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside 
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Clip<T>(min, max));

        result->addChild(input);
        return result;
    }

} // namespace nn::math
