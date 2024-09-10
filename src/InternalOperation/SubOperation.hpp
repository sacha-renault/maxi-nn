#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class SubOperation : public IOperation<T> {
    public:
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) override {
            // Ensure that there are exactly two columns (two children) for the subtraction
            if (children_values.cols() != 2) {
                throw std::invalid_argument("Sub operation requires exactly two children.");
            }

            // Perform element-wise subtraction of the two columns
            Eigen::Matrix<T, Eigen::Dynamic, 1> result = children_values.col(0) - children_values.col(1);
            return result;
        }

        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) override {
            // Ensure that there are exactly two columns (two children) for the subtraction
            if (children_values.cols() != 2) {
                throw std::invalid_argument("Sub operation requires exactly two children.");
            }

            // Create a matrix to store the gradients of the children
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_grads(children_values.rows(), 2);

            // Gradient for the first child is the same as the parent's gradient
            children_grads.col(0) = parent_grads;

            // Gradient for the second child is the negative of the parent's gradient
            children_grads.col(1) = -parent_grads;

            return children_grads;
        }
    };
} // namespace nn::Operation
