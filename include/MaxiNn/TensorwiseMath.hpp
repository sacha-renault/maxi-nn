#pragma once
#include <memory>
#include "../MaxiNn.hpp"

namespace nn::math
{
    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> reduceMean(std::shared_ptr<tensor::Tensor<T>> input) {
        auto mean_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input) -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
            // Compute the mean of all elements
            T mean_value = input.mean();
            Eigen::Matrix<T, Eigen::Dynamic, 1> result(1, 1);
            result(0, 0) = mean_value;
            return result;
        };

        auto mean_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input, const Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values, const Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads) -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
            // The gradient should be evenly distributed across all elements of the input tensor
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> grad_output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(input.rows(), input.cols(), parent_grads(0, 0) / input.size());
            return grad_output;
        };

        auto output_val = mean_forward(input->getValues());
        auto output = tensor::Tensor<T>::create({1}, output_val, std::make_shared<nn::Operation::TensorWiseOperation<T>>(mean_forward, mean_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> reduceSum(std::shared_ptr<tensor::Tensor<T>> input) {
        auto mean_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input) -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
            // Compute the mean of all elements
            T mean_value = input.sum();
            Eigen::Matrix<T, Eigen::Dynamic, 1> result(1, 1);
            result(0, 0) = mean_value;
            return result;
        };

        auto mean_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& input, const Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values, const Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads) -> Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> {
            // The gradient should be evenly distributed across all elements of the input tensor
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> grad_output = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(input.rows(), input.cols(), parent_grads(0, 0));
            return grad_output;
        };

        auto output_val = mean_forward(input->getValues());
        auto output = tensor::Tensor<T>::create({1}, output_val, std::make_shared<nn::Operation::TensorWiseOperation<T>>(mean_forward, mean_backward));
        output->addChild(input);
        return output;
    }
} // namespace nn::math
