#include <random>
#include <iostream>
#include "../include/MaxiNn.hpp"

namespace nn::tensor
{
    template <typename T>
    Tensor<T>::Tensor() : total_size_(1), values_(0){}

    template <typename T>
    Tensor<T>::Tensor(xt::dynamic_shape<size_t> dims, bool requires_grad)
        : dimensions_(dims), requires_grad_(requires_grad), stream_ptr(nullptr)
    {
        // Calculate total size
        total_size_ = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

        // Initialize values array with given dimensions
        values_ = xt::xarray<T>::from_shape(dims);

        // Initialize gradient array if required
        if (requires_grad_) {
            grads_ = xt::zeros<T>(dims);
        }
    }

    template <typename T>
    Tensor<T>::Tensor(xt::dynamic_shape<size_t> dims, xt::xarray<T> values, bool requires_grad)
        : Tensor<T>(dims, requires_grad)
    {
        values_ = values;  // Copy provided values
        stream_ptr = nullptr;  // No operation registered
    }

    template <typename T>
    Tensor<T>::Tensor(xt::dynamic_shape<size_t> dims, xt::xarray<T> values, std::shared_ptr<nn::Operation::IOperation<T>> stream, bool requires_grad)
        : Tensor<T>(dims, requires_grad)
    {
        values_ = values;  // Copy provided values
        stream_ptr = stream;  // Register operation
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::create() {
        return std::shared_ptr<Tensor<T>>(new Tensor<T>());
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::create(xt::dynamic_shape<size_t> dim, bool requires_grad) {
        return std::shared_ptr<Tensor<T>>(new Tensor<T>(dim, requires_grad));
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::create(xt::xarray<T> values, bool requires_grad) {
        return std::shared_ptr<Tensor<T>>(new Tensor<T>(values.shape(), values, requires_grad));
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::create(xt::xarray<T> values, std::shared_ptr<nn::Operation::IOperation<T>> stream, bool requires_grad) {
        return std::shared_ptr<Tensor<T>>(new Tensor<T>(values.shape(), values, stream, requires_grad));
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::zeros(xt::dynamic_shape<size_t> dims, bool requires_grad) {
        auto values = xt::zeros<T>(dims);
        auto new_tensor = Tensor::create(values, requires_grad);
        return new_tensor;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::ones(xt::dynamic_shape<size_t> dims, bool requires_grad) {
        auto values = xt::ones<T>(dims);
        auto new_tensor = Tensor::create(values, requires_grad);
        return new_tensor;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::random(xt::dynamic_shape<size_t> dims, T min, T max, bool requires_grad) {
        std::random_device rd;  // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_real_distribution<> dis(min, max); // Define the range

        xt::xarray<T> values = xt::empty<T>(dims);
        auto it = values.begin();
        while (it != values.end()) {
            *it++ = dis(gen);
        }
        auto new_tensor = Tensor::create(values, requires_grad);
        return new_tensor;
    }

    template <typename T>
    std::shared_ptr<Tensor<T>> Tensor<T>::normal(xt::dynamic_shape<size_t> dims, T mean, T stddev, bool requires_grad) {
        std::random_device rd;  // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::normal_distribution<> dis(mean, stddev); // Define the distribution

        xt::xarray<T> values = xt::empty<T>(dims);
        auto it = values.begin();
        while (it != values.end()) {
            *it++ = dis(gen);
        }
        auto new_tensor = Tensor::create(values, requires_grad);
        return new_tensor;
    }

    template <typename T>
    void Tensor<T>::display() const {
        std::cout << values_ << std::endl;
    }

    template <typename T>
    void Tensor<T>::displayGrad() const {
        std::cout << grads_ << std::endl;
    }

    template <typename T>
    T& Tensor<T>::operator[](const xt::xindex& idx) {
        return values_.element(idx.begin(), idx.end());;
    }

    template <typename T>
    T& Tensor<T>::getItem(const xt::xindex& idx) {
        return (*this)[idx];
    }

    template <typename T>
    const xt::xarray<T>& Tensor<T>::getValues() {
        return values_;
    }

    template <typename T>
    void Tensor<T>::setGrad(xt::xarray<T> grads) {
        if (grads.shape() == grads_.shape()) {
            grads_ = std::move(grads);
        } else {
            throw std::runtime_error("Cast is allowed only on grad accumulation");
        }
    }

    template <typename T>
    const xt::xarray<T>& Tensor<T>::getGrad() const {
        return grads_;
    }

    template <typename T>
    void Tensor<T>::resetGrad() {
        if (requires_grad_) {
            grads_.fill(T(0));
        }
    }

    template <typename T>
    void Tensor<T>::setOnesGrad() {
        if (requires_grad_) {
            grads_.fill(T(1));
        }
    }

    template <typename T>
    void Tensor<T>::addChild(std::shared_ptr<Tensor<T>> child) {
        children_.push_back(child);
    }

    template <typename T>
    const std::vector<std::shared_ptr<Tensor<T>>>& Tensor<T>::getChildren() const
    {
        return children_;
    };

    template <typename T>
    void Tensor<T>::setValues(xt::xarray<T> new_values) {
        if (values_.shape() == new_values.shape()) {
            values_ = std::move(new_values);
        } else {
            throw std::runtime_error("Cast is allowed only on grad accumulation");
        }
    }

    template <typename T>
    const xt::xarray<T>& Tensor<T>::getValues() const {
        return values_;
    }

    template <typename T>
    int Tensor<T>::size() const {
        return total_size_;
    }

    template <typename T>
    xt::dynamic_shape<size_t> Tensor<T>::shape() const {
        return values_.shape();
    }

    template <typename T>
    void Tensor<T>::fill(T value) {
        values_.fill(T(value));
    }

    template <typename T>
    void Tensor<T>::fill(xt::xarray<T> values) {
        values_ = std::move(values);
    }

    template <typename T>
    void Tensor<T>::setTensorType(TensorType type) {
        type_ = type;
    }

    template <typename T>
    void Tensor<T>::accumulateGrad(const xt::xarray<T>& add_grad) {
        if (grads_.shape() == add_grad.shape()) {
            grads_ += add_grad;
        } else {
            // Calculate the sum axes needed to accumulate add_grad
            xt::dynamic_shape<size_t> sum_axes;
            std::size_t dims = add_grad.shape().size();
            for (std::size_t i = 0; i < dims; ++i) {
                if (grads_.shape()[i] != add_grad.shape()[i]) {
                    sum_axes.push_back(i);
                }
            }

            // Sum over the axes where shapes don't match
            xt::xarray<T> reduced_add_grad = add_grad;
            for (int i = sum_axes.size() - 1 ; i >= 0 ; --i) {
                reduced_add_grad = xt::sum(reduced_add_grad, sum_axes[i]);
            }
            grads_ += xt::broadcast(reduced_add_grad, grads_.shape());
        }
    }

    template <typename T>
    void Tensor<T>::backward() {
        if (stream_ptr) {
            // Create an array to hold all children's values
            std::vector<std::reference_wrapper<const xt::xarray<T>>> children_values;

            // Point to actual children values
            for (const auto& child : children_) {
                children_values.emplace_back(child->values_);
            }

            // Compute the gradient
            std::vector<xt::xarray<T>> grads = stream_ptr->backward(children_values, values_, grads_);

            // Propagate the gradients to all the children
            for (int i = 0; i < children_.size(); ++i) {
                children_[i]->accumulateGrad(grads[i]);
            }
        }
    }



    template <typename T>
    void Tensor<T>::forward() {
        if (stream_ptr) {
            // Create an array to hold all children's values
            std::vector<std::reference_wrapper<const xt::xarray<T>>> children_values;

            // Point to actual children values
            for (const auto& child : children_) {
                children_values.emplace_back(child->values_);
            }

            // Compute the next values_
            values_ = stream_ptr->forward(children_values);
        }
    }


    // explicit instanciation of the class
    template class Tensor<float>;
    template class Tensor<double>;
}