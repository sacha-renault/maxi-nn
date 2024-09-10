#pragma once
#include "../MaxiNn.hpp"

using namespace nn::tensor;

template <typename T>
std::shared_ptr<Tensor<T>> operator+(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    if (lhs->getValues().size() != rhs->getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<int> dims = lhs->shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs->getValues() + rhs->getValues()).eval();
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::AddOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator*(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    if (lhs->getValues().size() != rhs->getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<int> dims = lhs->shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs->getValues().array() * rhs->getValues().array()).eval();
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::MulOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator-(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    if (lhs->getValues().size() != rhs->getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<int> dims = lhs->shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs->getValues().array() - rhs->getValues().array()).eval();
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::SubOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator/(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    if (lhs->getValues().size() != rhs->getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to divide.");
    }

    // Create a new Tensor for the result of the division
    std::vector<int> dims = lhs->shape();  // Make sure this is a vector
    Eigen::Matrix<T, Eigen::Dynamic, 1> valResult = (lhs->getValues().array() / rhs->getValues().array()).eval();
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::DivOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}
