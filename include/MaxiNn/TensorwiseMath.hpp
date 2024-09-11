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
        auto result = tensor::Tensor<T>::create(valResult.shape(), valResult, nn::Operation::Dot<T>);

        result->addChild(lt);
        result->addChild(rt);
        return result;
    }
//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> reduceMean(std::shared_ptr<tensor::Tensor<T>> input) {
//         auto mean_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
//             // Compute the mean of all elements
//             T mean_value = xt::mean(input)();
//             return xt::xarray<T>({mean_value});
//         };

//         auto mean_backward = [](const xt::xarray<T>& input, const xt::xarray<T>& parent_values, const xt::xarray<T>& parent_grads) -> xt::xarray<T> {
//             // The gradient should be evenly distributed across all elements of the input tensor
//             return xt::full_like(input, parent_grads(0) / input.size());
//         };

//         auto output_val = mean_forward(input->getValues());
//         auto output = tensor::Tensor<T>::create({1}, output_val, std::make_shared<nn::Operation::BatchWiseOperation<T>>(mean_forward, mean_backward));
//         output->addChild(input);
//         return output;
//     }

//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> reduceSum(std::shared_ptr<tensor::Tensor<T>> input) {
//         auto sum_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
//             // Compute the sum of all elements
//             T sum_value = xt::sum(input)();
//             return xt::xarray<T>({sum_value});
//         };

//         auto sum_backward = [](const xt::xarray<T>& input, const xt::xarray<T>& parent_values, const xt::xarray<T>& parent_grads) -> xt::xarray<T> {
//             // The gradient should be the same value distributed across all elements of the input tensor
//             return xt::full_like(input, parent_grads(0));
//         };

//         auto output_val = sum_forward(input->getValues());
//         auto output = tensor::Tensor<T>::create({1}, output_val, std::make_shared<nn::Operation::BatchWiseOperation<T>>(sum_forward, sum_backward));
//         output->addChild(input);
//         return output;
//     }

//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> reduceSum(std::shared_ptr<tensor::Tensor<T>> input, const std::vector<int>& axis) {
//         auto sum_forward = [axis](const xt::xarray<T>& input) -> xt::xarray<T> {
//             // Compute the sum along the specified axes
//             auto sum_value = xt::sum(input, axis);
//             return sum_value;
//         };

//         auto sum_backward = [axis](const xt::xarray<T>& input, const xt::xarray<T>& parent_values, const xt::xarray<T>& parent_grads) -> xt::xarray<T> {
//             // The gradient should be the same value distributed across all elements of the input tensor
//             xt::xarray<T> grad = xt::full_like(input, parent_grads(0));
//             return xt::broadcast(grad, input.shape());
//         };

//         auto output_val = sum_forward(input->getValues());

//         // Convert the shape to std::vector<size_t>
//         std::vector<size_t> shape(output_val.shape().begin(), output_val.shape().end());

//         // Assuming Tensor::create expects a shape and values
//         auto output = tensor::Tensor<T>::create(shape, output_val,
//             std::make_shared<nn::Operation::BatchWiseOperation<T>>(sum_forward, sum_backward));
//         output->addChild(input);
//         return output;
//     }

//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> dot(std::shared_ptr<tensor::Tensor<T>> a, std::shared_ptr<tensor::Tensor<T>> b) {
//         auto sum_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
//             // TODO
//         };

//         auto sum_backward = [axis](const xt::xarray<T>& input, const xt::xarray<T>& parent_values, const xt::xarray<T>& parent_grads) -> xt::xarray<T> {
//             // TODO
//         };

//         auto output_val = sum_forward(input->getValues());

//         // Convert the shape to std::vector<size_t>
//         std::vector<size_t> shape(output_val.shape().begin(), output_val.shape().end());

//         // Assuming Tensor::create expects a shape and values
//         auto output = tensor::Tensor<T>::create(shape, output_val,
//             std::make_shared<nn::Operation::BatchWiseOperation<T>>(sum_forward, sum_backward));
            
//         output->addChild(a);
//         output->addChild(b);
//         return output;
//     }
} // namespace nn::math
