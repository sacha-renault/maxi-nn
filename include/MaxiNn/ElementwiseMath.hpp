#pragma once
#include <memory>
#include "../MaxiNn.hpp"

namespace nn::math
{
    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> reLU(std::shared_ptr<tensor::Tensor<T>> input) {
        auto relu_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::maximum(input, T(0));  // ReLU: max(0, x)
        };

        auto relu_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::greater(input, T(0));  // Derivative of ReLU
        };

        auto output_val = relu_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(relu_forward, relu_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> tanh(std::shared_ptr<tensor::Tensor<T>> input) {
        auto tanh_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::tanh(input);
        };

        auto tanh_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            xt::xarray<T> tanh_val = xt::tanh(input);
            return T(1.0) - xt::square(tanh_val);
        };

        auto output_val = tanh_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(tanh_forward, tanh_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> pow(std::shared_ptr<tensor::Tensor<T>> input, T exponent) {
        auto pow_forward = [exponent](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::pow(input, exponent);
        };

        auto pow_backward = [exponent](const xt::xarray<T>& input) -> xt::xarray<T> {
            return exponent * xt::pow(input, exponent - T(1));
        };

        auto output_val = pow_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(pow_forward, pow_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> abs(std::shared_ptr<tensor::Tensor<T>> input) {
        auto abs_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::abs(input);
        };

        auto abs_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::sign(input);
        };

        auto output_val = abs_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(abs_forward, abs_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> exp(std::shared_ptr<tensor::Tensor<T>> input) {
        auto exp_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::exp(input);
        };

        auto exp_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::exp(input);  // Same as forward for exp
        };

        auto output_val = exp_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(exp_forward, exp_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> log(std::shared_ptr<tensor::Tensor<T>> input) {
        auto log_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::log(input);
        };

        auto log_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return T(1) / input;  // 1 / x
        };

        auto output_val = log_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(log_forward, log_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> sqrt(std::shared_ptr<tensor::Tensor<T>> input) {
        auto sqrt_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::sqrt(input);
        };

        auto sqrt_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return T(0.5) * xt::pow(input, T(-0.5));  // 1 / (2 * sqrt(x))
        };

        auto output_val = sqrt_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(sqrt_forward, sqrt_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> inverse(std::shared_ptr<tensor::Tensor<T>> input) {
        auto inverse_forward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return T(1) / input;
        };

        auto inverse_backward = [](const xt::xarray<T>& input) -> xt::xarray<T> {
            return -T(1) / xt::square(input);  // -1 / x^2
        };

        auto output_val = inverse_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(inverse_forward, inverse_backward));
        output->addChild(input);
        return output;
    }

    template <typename T>
    std::shared_ptr<tensor::Tensor<T>> clip(std::shared_ptr<tensor::Tensor<T>> input, T min_val, T max_val) {
        auto clip_forward = [min_val, max_val](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::clip(input, min_val, max_val);
        };

        auto clip_backward = [min_val, max_val](const xt::xarray<T>& input) -> xt::xarray<T> {
            return xt::cast<T>(input > min_val & input < max_val);
        };

        auto output_val = clip_forward(input->getValues());
        auto output = tensor::Tensor<T>::create(input->shape(), output_val, std::make_shared<Operation::ElementWiseOperation<T>>(clip_forward, clip_backward));
        output->addChild(input);
        return output;
    }

} // namespace nn::math
