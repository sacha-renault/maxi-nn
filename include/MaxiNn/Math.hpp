#pragma once
#include <memory>
#include "../MaxiNn.hpp"

namespace nn::math
{
    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> reLU(std::shared_ptr<tensor::Tensor<T>> input) {
        auto relu_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().max(0);  // ReLU: max(0, x)
        };

        auto relu_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return (input.array() > 0).template cast<T>();  // Derivative of ReLU
        };

        auto output_val = relu_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(relu_forward, relu_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> tanh(std::shared_ptr<tensor::Tensor<T>> input) {
        auto tanh_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().tanh();
        };

        auto tanh_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            Eigen::Matrix<T, Eigen::Dynamic, 1> tanh_val = input.array().tanh();
            return 1.0 - tanh_val.array().square();
        };

        auto output_val = tanh_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(tanh_forward, tanh_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> pow(std::shared_ptr<tensor::Tensor<T>> input, T exponent) {
        auto pow_forward = [exponent](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().pow(exponent);
        };

        auto pow_backward = [exponent](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return exponent * input.array().pow(exponent - 1);
        };

        auto output_val = pow_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(pow_forward, pow_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> abs(std::shared_ptr<tensor::Tensor<T>> input) {
        auto abs_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().abs();
        };

        auto abs_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().sign();
        };

        auto output_val = abs_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(abs_forward, abs_backward));
        output->addChild(input);
        return output;
    }
} // namespace nn::math
