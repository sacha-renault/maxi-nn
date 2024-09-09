#pragma once
#include "../../include/MaxiNn.hpp"
#include "Operation.hpp"

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
    Tensor<T> result(dims, valResult);
    result.addChild(lhs);
    result.addChild(rhs);
    result.setBackward(nn::Operation::addBackward<T>);
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
    Tensor<T> result(dims, valResult);
    result.addChild(lhs);
    result.addChild(rhs);
    result.setBackward(nn::Operation::mulBackward<T>);
    return result;
}

template <typename T>
Tensor<T> operator-(Tensor<T>& lhs, Tensor<T>& rhs);

template <typename T>
Tensor<T> operator/(Tensor<T>& lhs, Tensor<T>& rhs);