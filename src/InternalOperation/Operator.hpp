#pragma once
#include "../../include/MaxiNn.hpp"

using namespace nn::tensor;

template <typename T>
Tensor<T> operator+(Tensor<T>& lhs, Tensor<T>& rhs) {
    // Ensure the tensors have the same size
    if (lhs.getValues().size() != rhs.getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<int> dims = lhs.shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs.getValues() + rhs.getValues()).eval();
    Tensor<T> result(dims, valResult, std::make_shared<nn::Operation::AddOperation<T>>());
    result.addChild(lhs);
    result.addChild(rhs);
    return result;
}

template <typename T>
Tensor<T> operator*(Tensor<T>& lhs, Tensor<T>& rhs) {
    // Ensure the tensors have the same size
    if (lhs.getValues().size() != rhs.getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<int> dims = lhs.shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs.getValues().array() * rhs.getValues().array()).eval();
    Tensor<T> result(dims, valResult, std::make_shared<nn::Operation::MulOperation<T>>());
    result.addChild(lhs);
    result.addChild(rhs);
    return result;
}

template <typename T>
Tensor<T> operator-(Tensor<T>& lhs, Tensor<T>& rhs) {
    // Ensure the tensors have the same size
    if (lhs.getValues().size() != rhs.getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<int> dims = lhs.shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs.getValues().array() - rhs.getValues().array()).eval();
    Tensor<T> result(dims, valResult, std::make_shared<nn::Operation::SubOperation<T>>());
    result.addChild(lhs);
    result.addChild(rhs);
    return result;
}

template <typename T>
Tensor<T> operator/(Tensor<T>& lhs, Tensor<T>& rhs);