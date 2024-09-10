#pragma once
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class DivOperation : public IOperation<T> {
    public:
        // Forward pass: Element-wise division
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) override {
            if (children_values.cols() != 2) {
                throw std::invalid_argument("DivOperation requires exactly two child tensors.");
            }
            return (children_values.col(0).array() / children_values.col(1).array()).matrix();
        }

        // Backward pass: Gradient of the division
        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) override {
            if (children_values.cols() != 2) {
                throw std::invalid_argument("DivOperation requires exactly two child tensors.");
            }

            Eigen::Matrix<T, Eigen::Dynamic, 1> numerator = children_values.col(0);
            Eigen::Matrix<T, Eigen::Dynamic, 1> denominator = children_values.col(1);

            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> grads(children_values.rows(), 2);

            // Gradient with respect to the numerator (child 0)
            grads.col(0) = (parent_grads.array() / denominator.array()).matrix();

            // Gradient with respect to the denominator (child 1)
            grads.col(1) = (-parent_grads.array() * numerator.array() / denominator.array().square()).matrix();

            return grads;
        }
    };
} // namespace nn::Operation
