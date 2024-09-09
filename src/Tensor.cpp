#include "../include/MaxiNn.hpp"
#include <iostream>

namespace nn::tensor
{ 
    template <typename T>
    Tensor<T>::Tensor() : total_size_(1), values_(0){}

    template <typename T>
    Tensor<T>::Tensor(std::vector<int> dims, bool requires_grad) : total_size_(1), requires_grad_(requires_grad) {
        for(int i = 0 ; i < dims.size() ; ++i) {
            total_size_ *= dims[i];
        }

        // init values with empty array
        values_ = Eigen::Matrix<T, Eigen::Dynamic, 1>(total_size_);

        // copy dimension into internal 
        dimensions_ = dims;

        // only allocate grad if required
        if (requires_grad_) {
            grads_ = Eigen::Matrix<T, Eigen::Dynamic, 1>(total_size_);
            grads_.setConstant(0);
        }
    }

    template <typename T>
    Tensor<T>::Tensor(std::vector<int> dim, Eigen::Matrix<T, Eigen::Dynamic, 1> values, bool requires_grad) 
        : Tensor<T>(dim, requires_grad) {
        values_ = values; // copy values
    }

    template <typename T>
    int Tensor<T>::computeIndex(const std::vector<int>& multi_dim_index) const {
        int index = 0;
        int multiplier = 1;
        if (multi_dim_index.size() != dimensions_.size()) {
            throw std::runtime_error("Cannot get the index with indices size != rank of the tensor");
        }
        for (int i = dimensions_.size() - 1; i >= 0; --i) {
            index += multi_dim_index[i] * multiplier;
            multiplier *= dimensions_[i];
        }
        return index;
    }

    template <typename T>
    T& Tensor<T>::operator[](const std::vector<int>& multi_dim_index) {
        return values_(computeIndex(multi_dim_index));
    }

    template <typename T>
    void Tensor<T>::resetGrad() {
        if (requires_grad_) {
            grads_.setConstant(0);
        }
    }

    template <typename T>
    void Tensor<T>::setOnesGrad() {
        if (requires_grad_) {
            grads_.setConstant(1);
        }
    }

    template <typename T>
    void Tensor<T>::addChild(const Tensor<T>& child) {
        children_.push_back(std::make_shared<Tensor>(child));
    }

    template <typename T>
    void Tensor<T>::setValues(Eigen::Matrix<T, Eigen::Dynamic, 1> new_values) {
        values_ = new_values; // copy into values_
    }

    template <typename T>
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& Tensor<T>::getValues() const {
        return values_;
    }

    template <typename T>
    int Tensor<T>::size() const {
        return dimensions_.size();
    }

    template <typename T>
    std::vector<int> Tensor<T>::shape() const {
        return dimensions_;
    }

    template <typename T>
    void Tensor<T>::fill(T value) {
        values_.setConstant(value);
    }

    template <typename T>
    void Tensor<T>::accumulateGrad(const Eigen::Matrix<T, Eigen::Dynamic, 1>& add_grad) {
        if (requires_grad_) {
            if (grads_.size() == add_grad.size()) {
            grads_ += add_grad;
            } else {
                throw std::runtime_error("Gradient dimensions do not match.");
            }
        }
    }

    template <typename T>
    void Tensor<T>::backward() {
        if (backward_) {
            int numChildren = children_.size();
            int numRows = values_.rows();

            // Matrix to store mapped columns
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> childrenValues(numRows, numChildren);

            // Use Eigen::Map to reference each child's values without copying
            for (int i = 0; i < numChildren; ++i) {
                Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> childMap(children_[i]->values_.data(), numRows);
                childrenValues.col(i) = childMap;
            }

            // compute the gradient
            auto grads = backward_(childrenValues, values_, grads_);

            // Propagate the gradients to each child
            for (int i = 0; i < numChildren; ++i) {
                children_[i]->grads_ += grads.col(i);
            }

            // show grads for every childs
            for (int i = 0; i < numChildren; ++i) {
                std::cout << "Child n°" << i << std::endl;
                for (int j = 0 ; j < children_[i]->size() ; ++j) {
                    std::cout << children_[i]->grads_(j) << std::endl;
                }
            }
        }
    }

    template <typename T>
    void Tensor<T>::setBackward(nn::Operation::BackwardFunc<T> func) {
        backward_ = func;
    }

    // explicit instanciation of the class
    template class Tensor<float>;
}