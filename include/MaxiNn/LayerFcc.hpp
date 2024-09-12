#pragma once
#include "../MaxiNn.hpp"

namespace nn::layers
{
    template <typename T>
    class Fcc {
    public:
        Fcc(size_t num_input, size_t num_output) {
            // default initialize with Xavier
            T limit = std::sqrt(6.0 / (num_input + num_output));
            weights_ = tensor::Tensor<T>::random({num_input, num_output}, -limit, limit);
            bias_ = tensor::Tensor<T>::zeros({1, num_output});
        }

        Fcc(size_t num_input, size_t num_output, std::function<std::shared_ptr<tensor::Tensor<T>>(std::shared_ptr<tensor::Tensor<T>>)> activation)
            : Fcc(num_input, num_output) { activation_ = activation; }

        std::shared_ptr<tensor::Tensor<T>> operator()(std::shared_ptr<tensor::Tensor<T>> input) {
            // ensure rank is two (bs, num_inputs)
            if (input->shape().size() != 2) {
                throw std::runtime_error("Fcc layer accept input with shape (bs, input_size)");
            }

            // compute forward
            auto out = nn::math::dot(input, weights_) + bias_;

            // eventual activation
            if (activation_) {
                out = activation_(out);
            }

            return out;
        }

        std::shared_ptr<tensor::Tensor<T>> getWeights() {
            return weights_;
        }

        std::shared_ptr<tensor::Tensor<T>> getBias() {
            return bias_;
        }
    private:
        std::shared_ptr<tensor::Tensor<T>> weights_;
        std::shared_ptr<tensor::Tensor<T>> bias_;
        std::function<std::shared_ptr<tensor::Tensor<T>>(std::shared_ptr<tensor::Tensor<T>>)> activation_;
    };

    using FFcc = Fcc<float>;

} // namespace nn::layers
