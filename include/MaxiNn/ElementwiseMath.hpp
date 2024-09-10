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

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> exp(std::shared_ptr<tensor::Tensor<T>> input) {
        auto exp_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().exp();
        };

        auto exp_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().exp();  // Same as forward for exp
        };

        auto output_val = exp_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(exp_forward, exp_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> log(std::shared_ptr<tensor::Tensor<T>> input) {
        auto log_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().log();
        };

        auto log_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().inverse();  // 1 / x
        };

        auto output_val = log_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(log_forward, log_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> sqrt(std::shared_ptr<tensor::Tensor<T>> input) {
        auto sqrt_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().sqrt();
        };

        auto sqrt_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return 0.5 * input.array().inverse().sqrt();  // 1 / (2 * sqrt(x))
        };

        auto output_val = sqrt_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(sqrt_forward, sqrt_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> inverse(std::shared_ptr<tensor::Tensor<T>> input) {
        auto inverse_forward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().inverse();
        };

        auto inverse_backward = [](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return -input.array().inverse().square();  // -1 / x^2
        };

        auto output_val = inverse_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(inverse_forward, inverse_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> clip(std::shared_ptr<tensor::Tensor<T>> input, T min_val, T max_val) {
        auto clip_forward = [min_val, max_val](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return input.array().min(max_val).max(min_val);
        };

        auto clip_backward = [min_val, max_val](const Eigen::Matrix<T, Eigen::Dynamic, 1>& input) -> Eigen::Matrix<T, Eigen::Dynamic, 1> {
            return ((input.array() > min_val) && (input.array() < max_val)).template cast<T>();
        };

        auto output_val = clip_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(clip_forward, clip_backward));
        output->addChild(input);
        return output;
    }

} // namespace nn::math
