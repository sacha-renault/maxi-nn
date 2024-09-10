#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class AddOperation : public IOperation<T> {
    public:
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) override {
            // element wise sum
            Eigen::Matrix<T, Eigen::Dynamic, 1> result = children_values.rowwise().sum();
            return result;
        }

        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) override {
            // Create a matrix to store the gradients of the children
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> childrenGrads(children_values.rows(), children_values.cols());

            // Propagate the parent's gradient to each child
            for (int i = 0; i < children_values.cols(); ++i) {
                childrenGrads.col(i) = parent_grads;  // For add, the gradient is simply passed through
            }

            return childrenGrads;
        }
    };
} // namespace nn::Operation
