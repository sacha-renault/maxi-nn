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
        stream_ptr = nullptr; // no registered operation
    }

    template <typename T>
    Tensor<T>::Tensor(std::vector<int> dim, Eigen::Matrix<T, Eigen::Dynamic, 1> values, std::shared_ptr<nn::Operation::IOperation<T>> stream, bool requires_grad)
        : Tensor<T>(dim, requires_grad) {
        values_ = values; // copy values
        stream_ptr = stream;
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
    void Tensor<T>::displayInternal(const Eigen::Matrix<T, Eigen::Dynamic, 1>& displayable) const {
        int numDims = dimensions_.size();

        if (numDims == 1) {
            std::cout << "[ ";
            for (int i = 0; i < displayable.rows(); ++i) {
                std::cout << displayable(i, 0);
                if (i < displayable.rows() - 1) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        } else if (numDims == 2) {
            std::cout << "[" << std::endl;
            for (int i = 0; i < displayable.rows(); ++i) {
                std::cout << "    [ ";
                for (int j = 0; j < displayable.cols(); ++j) {
                    std::cout << displayable(i, j);
                    if (j < displayable.cols() - 1) std::cout << ", ";
                }
                std::cout << " ]" << std::endl;
            }
            std::cout << "]" << std::endl;
        } else {
            // For 3D or higher dimensions, we'll recursively print the slices.
            std::function<void(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&, int)> displayRecursively =
                [&](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& matrix, int dimIndex) {
                    if (dimIndex == numDims - 2) {
                        std::cout << "    [";
                        for (int i = 0; i < matrix.rows(); ++i) {
                            std::cout << std::endl << "        [ ";
                            for (int j = 0; j < matrix.cols(); ++j) {
                                std::cout << matrix(i, j);
                                if (j < matrix.cols() - 1) std::cout << ", ";
                            }
                            std::cout << " ]";
                        }
                        std::cout << std::endl << "    ]";
                    } else {
                        std::cout << "[" << std::endl;
                        for (int i = 0; i < matrix.rows(); ++i) {
                            std::cout << "    ";
                            displayRecursively(matrix, dimIndex + 1);
                            if (i < matrix.rows() - 1) std::cout << ",";
                            std::cout << std::endl;
                        }
                        std::cout << "]";
                    }
                };

            std::cout << "[" << std::endl;
            displayRecursively(displayable, 0);
            std::cout << std::endl << "]" << std::endl;
        }
    }

    template <typename T>
    void Tensor<T>::display() const {
        this->displayInternal(values_);
    }

    template <typename T>
    void Tensor<T>::displayGrad() const {
        this->displayInternal(grads_);
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
        if (stream_ptr) {
            int numChildren = children_.size();
            int numRows = values_.rows();

            // Matrix to store mapped columns
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values(numRows, numChildren);

            // Use Eigen::Map to reference each child's values. No cpy
            for (int i = 0; i < numChildren; ++i) {
                Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> childMap(children_[i]->values_.data(), numRows);
                children_values.col(i) = childMap;
            }

            // compute the gradient
            auto grads = stream_ptr->backward(children_values, values_, grads_);

            // Propagate the gradients to all the children
            for (int i = 0; i < numChildren; ++i) {
                children_[i]->grads_ += grads.col(i);
            }

            // // show grads for every chldren
            // for (int i = 0; i < numChildren; ++i) {
            //     std::cout << "Child n°" << i + 1 << std::endl;
            //     for (int j = 0 ; j < children_[i]->size() ; ++j) {
            //         std::cout << children_[i]->grads_(j) << std::endl;
            //     }
            // }
        }
    }

    template <typename T>
    void Tensor<T>::forward() {
        if (stream_ptr) {
            int numChildren = children_.size();
            int numRows = values_.rows();

            // Matrix to store mapped columns
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values(numRows, numChildren);

            // Use Eigen::Map to reference each child's values. No cpy
            for (int i = 0; i < numChildren; ++i) {
                Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> childMap(children_[i]->values_.data(), numRows);
                children_values.col(i) = childMap;
            }

            // compute the next values_
            values_ = stream_ptr->forward(children_values);
        }
    }

    // explicit instanciation of the class
    template class Tensor<float>;
}