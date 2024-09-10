#pragma once
#include <functional>
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class TensorWiseOperation : public IOperation<T> {
    private:
        // Forward function for the entire tensor or specific dimensions
        std::function<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&)> forward_func;

        // Backward function for the entire tensor or specific dimensions
        std::function<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&, const Eigen::Matrix<T, Eigen::Dynamic, 1>&, const Eigen::Matrix<T, Eigen::Dynamic, 1>&)> backward_func;

    public:
        // Forward pass
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) override {
            return forward_func(children_values);
        }

        // Backward pass
        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) override {
            return backward_func(children_values, parent_values, parent_grads);
        }

        // Constructor to set the forward and backward functions
        TensorWiseOperation(
            std::function<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&)> f,
            std::function<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>&, const Eigen::Matrix<T, Eigen::Dynamic, 1>&, const Eigen::Matrix<T, Eigen::Dynamic, 1>&)> b
        ) : forward_func(f), backward_func(b) { }
    };
} // namespace nn::Operation
