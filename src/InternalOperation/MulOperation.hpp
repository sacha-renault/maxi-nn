#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class MulOperation : public IOperation<T> {
    public:
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) override {
            // element wise sum
            Eigen::Matrix<T, Eigen::Dynamic, 1> result = children_values.rowwise().prod();
            return result;
        }

        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) override {
            // Check that there are exactly two columns (two children)
            if (children_values.cols() != 2) {
                throw std::invalid_argument("Mul operation requires exactly two children.");
            }

            // Create a matrix to store the gradients of the children
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_grads(children_values.rows(), 2);

            // Calculate the gradient for each child using the chain rule
            // Grad for child 1 is parent_grads * value of child 2
            children_grads.col(0) = parent_grads.array() * children_values.col(1).array();

            // Grad for child 2 is parent_grads * value of child 1
            children_grads.col(1) = parent_grads.array() * children_values.col(0).array();

            return children_grads;
        }
    };
} // namespace nn::Operation
