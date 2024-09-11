#pragma once
#include "../MaxiNn.hpp"

using namespace nn::tensor;

template <typename T>
std::shared_ptr<Tensor<T>> operator+(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    // if (lhs->getValues().size() != rhs->getValues().size()) {
    //     throw std::runtime_error("Tensors must have the same size to add.");
    // }

    // Create a new Tensor for the result of the addition
    std::vector<std::size_t> dims = lhs->shape();
    xt::xarray<T> valResult = (lhs->getValues() + rhs->getValues());
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::AddOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator*(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    // if (lhs->getValues().size() != rhs->getValues().size()) {
    //     throw std::runtime_error("Tensors must have the same size to add.");
    // }

    // Create a new Tensor for the result of the addition
    std::vector<std::size_t> dims = lhs->shape();
    xt::xarray<T> valResult = (lhs->getValues() * rhs->getValues());
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::MulOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator*(std::shared_ptr<Tensor<T>> lhs, T rhst) {
    // Get the dimensions of the lhs tensor
    std::vector<std::size_t> dims = lhs->shape();

    // Create a new Tensor for the scalar value
    auto rhs = Tensor<T>::create(dims);
    rhs->fill(rhst);  // Fill rhs tensor with the scalar value

    // Perform element-wise multiplication
    xt::xarray<T> valResult = (lhs->getValues() * rhs->getValues());

    // Create the result tensor with the computed values
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::MulOperation<T>>());

    // Add lhs and rhs as children to the result tensor
    result->addChild(lhs);
    result->addChild(rhs);

    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator*(T lhst, std::shared_ptr<Tensor<T>> rhs) {
    return operator*<T>(rhs, lhst);
}

template <typename T>
std::shared_ptr<Tensor<T>> operator-(std::shared_ptr<Tensor<T>> lhs, std::shared_ptr<Tensor<T>> rhs) {
    // Ensure the tensors have the same size
    if (lhs->getValues().size() != rhs->getValues().size()) {
        throw std::runtime_error("Tensors must have the same size to add.");
    }

    // Create a new Tensor for the result of the addition
    std::vector<std::size_t> dims = lhs->shape();
    xt::xarray<T> valResult = (lhs->getValues() - rhs->getValues());
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
    std::vector<std::size_t> dims = lhs->shape();
    xt::xarray<T> valResult = (lhs->getValues() / rhs->getValues());
    std::shared_ptr<Tensor<T>> result = Tensor<T>::create(dims, valResult, std::make_shared<nn::Operation::DivOperation<T>>());
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<Tensor<T>> operator/(std::shared_ptr<Tensor<T>> lhs, T rhst) {
    return operator*<T>(lhs, 1 / rhst);
}

