// #pragma once
// #include "../../include/MaxiNn.hpp"
// #include <memory>

// namespace nn::neuron
// {
//     template <typename T>
//     class Neuron {
//     public:
//         Neuron(size_t num_inputs);

//         // Forward pass: Calculate the output of the neuron
//         std::shared_ptr<tensor::Tensor<T>> forward(std::shared_ptr<tensor::Tensor<T>> inputs);

//         // Setters and Getters for weights and bias
//         void setWeights(std::shared_ptr<tensor::Tensor<T>> new_weights);
//         std::shared_ptr<tensor::Tensor<T>> getWeights() const;

//         void setBias(std::shared_ptr<tensor::Tensor<T>> new_bias);
//         std::shared_ptr<tensor::Tensor<T>> getBias() const;

//     private:
//         std::shared_ptr<tensor::Tensor<T>> weights_;
//         std::shared_ptr<tensor::Tensor<T>> bias_;
//     };

//     // Constructor
//     template <typename T>
//     Neuron<T>::Neuron(size_t num_inputs) {
//         weights_ = tensor::Tensor<T>::random({1, num_inputs});
//         bias_ = tensor::Tensor<T>::random({1});  // Initialize bias as a tensor::Tensor with shape (1, 1)
//     }

//     // Forward pass
//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> Neuron<T>::forward(std::shared_ptr<tensor::Tensor<T>> inputs) {
//         return nn::math::reduceSum(inputs * weights_ , { 1 }) + bias_;
//     }

//     // Set weights
//     template <typename T>
//     void Neuron<T>::setWeights(std::shared_ptr<tensor::Tensor<T>> new_weights) {
//         weights_ = new_weights;
//     }

//     // Get weights
//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> Neuron<T>::getWeights() const {
//         return weights_;
//     }

//     // Set bias
//     template <typename T>
//     void Neuron<T>::setBias(std::shared_ptr<tensor::Tensor<T>> new_bias) {
//         bias_ = new_bias;
//     }

//     // Get bias
//     template <typename T>
//     std::shared_ptr<tensor::Tensor<T>> Neuron<T>::getBias() const {
//         return bias_;
//     }

// } // namespace nn::neuron
