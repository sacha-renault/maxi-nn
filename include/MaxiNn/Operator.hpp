#pragma once
#include "../MaxiNn.hpp"

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator+(std::shared_ptr<nn::tensor::Tensor<T>> lhs, std::shared_ptr<nn::tensor::Tensor<T>> rhs) {
    // First make a forward pass with the operation
    xt::xarray<T> valResult = nn::Operation::Add2<T>->forward({lhs->getValues(), rhs->getValues()});

    // get the result and create a tensor with same shape -> set the result data inside
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Add2<T>);

    // add the two children in the result
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator+(std::shared_ptr<nn::tensor::Tensor<T>> lhs, T scalar) {
    // Create a new Tensor for the scalar value
    auto rhs = nn::tensor::Tensor<T>::create(lhs->shape());
    rhs->fill(scalar);  // Fill rhs tensor with the scalar value

    // First make a forward pass with the operation
    xt::xarray<T> valResult = nn::Operation::Add2<T>->forward({lhs->getValues(), rhs->getValues()});

    // get the result and create a tensor with same shape -> set the result data inside
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Add2<T>);

    // add the two children in the result
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator*(std::shared_ptr<nn::tensor::Tensor<T>> lhs, std::shared_ptr<nn::tensor::Tensor<T>> rhs) {
    // First make a forward pass with the operation
    xt::xarray<T> valResult = nn::Operation::Mul2<T>->forward({lhs->getValues(), rhs->getValues()});

    // get the result and create a tensor with same shape -> set the result data inside
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Mul2<T>);

    // add the two children in the result
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator*(std::shared_ptr<nn::tensor::Tensor<T>> lhs, T rhst) {

    // Create a new Tensor for the scalar value
    auto rhs = nn::tensor::Tensor<T>::create(lhs->shape());
    rhs->fill(rhst);  // Fill rhs tensor with the scalar value

    // Perform element-wise multiplication
    xt::xarray<T> valResult = nn::Operation::Mul2<T>->forward({lhs->getValues(), rhs->getValues()});

    // Create the result tensor with the computed values
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Mul2<T>);

    // Add lhs and rhs as children to the result tensor
    result->addChild(lhs);
    result->addChild(rhs);

    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator*(T lhst, std::shared_ptr<nn::tensor::Tensor<T>> rhs) {
    return operator*<T>(rhs, lhst);
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator-(std::shared_ptr<nn::tensor::Tensor<T>> lhs, std::shared_ptr<nn::tensor::Tensor<T>> rhs) {
    // First make a forward pass with the operation
    xt::xarray<T> valResult = nn::Operation::Sub2<T>->forward({lhs->getValues(), rhs->getValues()});

    // get the result and create a tensor with same shape -> set the result data inside
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Sub2<T>);

    // add the two children in the result
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator/(std::shared_ptr<nn::tensor::Tensor<T>> lhs, std::shared_ptr<nn::tensor::Tensor<T>> rhs) {
    // First make a forward pass with the operation
    xt::xarray<T> valResult = nn::Operation::Div2<T>->forward({lhs->getValues(), rhs->getValues()});

    // get the result and create a tensor with same shape -> set the result data inside
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Div2<T>);

    // add the two children in the result
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator/(T lhst, std::shared_ptr<nn::tensor::Tensor<T>> rhs) {
    // Create a new Tensor for the scalar value
    auto lhs = nn::tensor::Tensor<T>::create(rhs->shape());
    lhs->fill(lhst);  // Fill lhs tensor with the scalar value

    // First make a forward pass with the operation
    xt::xarray<T> valResult = nn::Operation::Div2<T>->forward({lhs->getValues(), rhs->getValues()});

    // get the result and create a tensor with same shape -> set the result data inside
    std::shared_ptr<nn::tensor::Tensor<T>> result = nn::tensor::Tensor<T>::create(valResult, nn::Operation::Div2<T>);

    // add the two children in the result
    result->addChild(lhs);
    result->addChild(rhs);
    return result;
}

template <typename T>
std::shared_ptr<nn::tensor::Tensor<T>> operator/(std::shared_ptr<nn::tensor::Tensor<T>> lhs, T rhst) {
    return operator*<T>(lhs, 1 / rhst);
}

