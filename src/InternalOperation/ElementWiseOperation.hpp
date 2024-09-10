#pragma once
#include <functional>
#include "InternalOperation.hpp"

namespace nn::Operation
{
    template <typename T>
    class ElementWiseOperation : public IOperation<T> {
    private:
        std::function<Eigen::Matrix<T, Eigen::Dynamic, 1>(const Eigen::Matrix<T, Eigen::Dynamic, 1>&)> forward_func;
        std::function<Eigen::Matrix<T, Eigen::Dynamic, 1>(const Eigen::Matrix<T, Eigen::Dynamic, 1>&)> backward_func;
    public:
        virtual Eigen::Matrix<T, Eigen::Dynamic, 1> forward(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> children_values) override {
            if (children_values.cols() != 1) {
                throw std::invalid_argument("Element-wise operation must have exactly one child tensor.");
            }

            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> single_child(children_values.col(0).data(), children_values.rows());
            return forward_func(single_child);
        }

        virtual Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> backward(
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& children_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_values,
            Eigen::Matrix<T, Eigen::Dynamic, 1>& parent_grads
        ) override {
            if (children_values.cols() != 1) {
                throw std::invalid_argument("Element-wise operation must have exactly one child tensor.");
            }

            Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> single_child(children_values.col(0).data(), children_values.rows());

            auto local_grads = backward_func(single_child);

            Eigen::Matrix<T, Eigen::Dynamic, 1> children_grads = local_grads.array() * parent_grads.array();

            return children_grads;
        }

        ElementWiseOperation(
            std::function<Eigen::Matrix<T, Eigen::Dynamic, 1>(const Eigen::Matrix<T, Eigen::Dynamic, 1>&)> f,
            std::function<Eigen::Matrix<T, Eigen::Dynamic, 1>(const Eigen::Matrix<T, Eigen::Dynamic, 1>&)> b
        ) : forward_func(f), backward_func(b) { }
    };
} // namespace nn::Operation

