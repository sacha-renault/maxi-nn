#pragma once
#include <functional>
#include <eigen3/Eigen/Dense>
#include <vector>

namespace nn::Operation
{
    template <typename T>
    class IOperation {
    public:
        virtual ~IOperation() = default;

        /// @brief Compute the values for output tensor from children values
        /// @param children_values
        /// @return values to set for output node
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) = 0;

        /// @brief Compute the gradient to accumulate for children
        /// @param children_values values of children
        /// @param parent_values values of parent
        /// @param parent_grads grad of parent
        /// @return a matrix containing gradient to accumulate on children nodes
        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) = 0;
    };

    // template <typename T>
    // using BackwardFunc = std::function<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> (
    //     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&, // all the values of children
    //     Eigen::Matrix<T, Eigen::Dynamic, 1>&,              // values of parent
    //     Eigen::Matrix<T, Eigen::Dynamic, 1>&               // grad of parent
    // )>;

    // // Backward function for the Add operation
    // template <typename T>
    // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> addBackward(
    //     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& childrenValues,
    //     Eigen::Matrix<T, Eigen::Dynamic, 1>& parentValues,
    //     Eigen::Matrix<T, Eigen::Dynamic, 1>& parentGrad
    // ) {
    //     // Create a matrix to store the gradients of the children
    //     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> childrenGrads(childrenValues.rows(), childrenValues.cols());

    //     // Propagate the parent's gradient to each child
    //     for (int i = 0; i < childrenValues.cols(); ++i) {
    //         childrenGrads.col(i) = parentGrad;  // For add, the gradient is simply passed through
    //     }

    //     return childrenGrads;
    // }

    // // Backward function for the Mul operation
    // template <typename T>
    // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mulBackward(
    //     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& childrenValues,
    //     Eigen::Matrix<T, Eigen::Dynamic, 1>& parentValues,
    //     Eigen::Matrix<T, Eigen::Dynamic, 1>& parentGrad
    // ) {
    //     // Check that there are exactly two columns (two children)
    //     if (childrenValues.cols() != 2) {
    //         throw std::invalid_argument("Mul operation requires exactly two children.");
    //     }

    //     // Create a matrix to store the gradients of the children
    //     Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> childrenGrads(childrenValues.rows(), 2);

    //     // Calculate the gradient for each child using the chain rule
    //     // Grad for child 1 is parentGrad * value of child 2
    //     childrenGrads.col(0) = parentGrad.array() * childrenValues.col(1).array();

    //     // Grad for child 2 is parentGrad * value of child 1
    //     childrenGrads.col(1) = parentGrad.array() * childrenValues.col(0).array();

    //     return childrenGrads;
    // }
} // namespace nn::Operation
