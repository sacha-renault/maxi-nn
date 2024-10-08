#pragma once
#include <memory>
#include "../MaxiNn.hpp"

namespace nn::math
{
    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> dot(std::shared_ptr<tensor::Tensor<T>> lt, std::shared_ptr<tensor::Tensor<T>> rt) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::Dot<T>->forward({lt->getValues(), rt->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside
        auto result = tensor::Tensor<T>::create(valResult, nn::Operation::Dot<T>);

        result->addChild(lt);
        result->addChild(rt);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> reduceSum(std::shared_ptr<tensor::Tensor<T>> input, xt::dynamic_shape<size_t> axis = xt::dynamic_shape<size_t>()) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::ReduceSum<T>(axis)->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside
        auto result = tensor::Tensor<T>::create(valResult, nn::Operation::ReduceSum<T>(axis));

        result->addChild(input);
        return result;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> reduceMean(std::shared_ptr<tensor::Tensor<T>> input, xt::dynamic_shape<size_t> axis = xt::dynamic_shape<size_t>()) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::ReduceMean<T>(axis)->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside
        auto result = tensor::Tensor<T>::create(valResult, nn::Operation::ReduceMean<T>(axis));

        result->addChild(input);
        return result;
    }

    /// @brief softmax funtion on dimension 1 (expect tensor of shape (batch_size, num_inputs))
    /// @param input tensor
    /// @return softmaxed tensor
    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> softmax(std::shared_ptr<tensor::Tensor<T>> input) {
        // First make a forward pass with the operation
        xt::xarray<T> valResult = nn::Operation::SoftmaxDim1<T>->forward({input->getValues()});

        // get the result and create a tensor with same shape -> set the result data inside
        auto result = tensor::Tensor<T>::create(valResult, nn::Operation::SoftmaxDim1<T>);

        result->addChild(input);
        return result;
    }
} // namespace nn::math
