#pragma once
#include "../../include/MaxiNn.hpp"
#include <memory>

namespace nn::neuron
{

    template <typename T>
    class Neuron {
    public:
        Neuron(int num_inputs);

        // Forward pass: Calculate the output of the neuron
        std::shared_ptr<Tensor<T>> forward(std::shared_ptr<Tensor<T>> inputs);

        // Setters and Getters for weights and bias
        void setWeights(std::shared_ptr<Tensor<T>> new_weights);
        std::shared_ptr<Tensor<T>> getWeights() const;

        void setBias(std::shared_ptr<Tensor<T>> new_bias);
        std::shared_ptr<Tensor<T>> getBias() const;

    private:
        std::shared_ptr<Tensor<T>> weights_;
        std::shared_ptr<Tensor<T>> bias_;

        // Activation function (e.g., sigmoid)
        std::shared_ptr<Tensor<T>> sigmoid(std::shared_ptr<Tensor<T>> x) const;

        // Utility function to initialize weights
        void initializeWeights(int num_inputs);
    };

    // Constructor
    template <typename T>
    Neuron<T>::Neuron(int num_inputs) {
        initializeWeights(num_inputs);
        bias_ = Tensor<T>::zeros({1, 1});  // Initialize bias as a Tensor with shape (1, 1)
    }

    // Forward pass
    template <typename T>
    std::shared_ptr<Tensor<T>> Neuron<T>::forward(std::shared_ptr<Tensor<T>> inputs) {
        // Weighted sum: input * weights + bias
        auto weighted_sum = inputs->matmul(weights_)->add(bias_);

        // Apply activation function (e.g., sigmoid)
        return sigmoid(weighted_sum);
    }

    // Sigmoid activation function
    template <typename T>
    std::shared_ptr<Tensor<T>> Neuron<T>::sigmoid(std::shared_ptr<Tensor<T>> x) const {
        // Assuming the Tensor class has an element-wise operation method
        return x->apply([](T val) { return 1.0 / (1.0 + std::exp(-val)); });
    }

    // Initialize weights randomly
    template <typename T>
    void Neuron<T>::initializeWeights(int num_inputs) {
        weights_ = Tensor<T>::random({num_inputs, 1}, -1.0, 1.0);  // Random initialization in range [-1, 1]
    }

    // Set weights
    template <typename T>
    void Neuron<T>::setWeights(std::shared_ptr<Tensor<T>> new_weights) {
        weights_ = new_weights;
    }

    // Get weights
    template <typename T>
    std::shared_ptr<Tensor<T>> Neuron<T>::getWeights() const {
        return weights_;
    }

    // Set bias
    template <typename T>
    void Neuron<T>::setBias(std::shared_ptr<Tensor<T>> new_bias) {
        bias_ = new_bias;
    }

    // Get bias
    template <typename T>
    std::shared_ptr<Tensor<T>> Neuron<T>::getBias() const {
        return bias_;
    }

} // namespace nn::neuron
