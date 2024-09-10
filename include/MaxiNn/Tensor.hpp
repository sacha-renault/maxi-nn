#pragma once
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include "../../src/InternalOperation/InternalOperation.hpp"

namespace nn::tensor
{
    template <typename T>
    class Tensor{
    protected:
        // values and grads
        Eigen::Matrix<T, Eigen::Dynamic, 1> values_;
        Eigen::Matrix<T, Eigen::Dynamic, 1> grads_;

        // bools
        bool requires_grad_;

        // dimensions
        std::vector<int> dimensions_;
        int total_size_;

        // children
        std::vector<std::shared_ptr<Tensor>> children_;

        // backward function
        std::shared_ptr<nn::Operation::IOperation<T>> stream_ptr;

        // private functions
        int computeIndex(const std::vector<int>& multi_dim_index) const;
        void displayInternal(const Eigen::Matrix<T, Eigen::Dynamic, 1>& displayable) const;
    public:
        Tensor();
        Tensor(std::vector<int> dim, bool requires_grad = true);
        Tensor(std::vector<int> dim, Eigen::Matrix<T, Eigen::Dynamic, 1> values, bool requires_grad = true);
        Tensor(std::vector<int> dim, Eigen::Matrix<T, Eigen::Dynamic, 1> values, std::shared_ptr<nn::Operation::IOperation<T>> stream, bool requires_grad = true);

        // gradient
        void accumulateGrad(const Eigen::Matrix<T, Eigen::Dynamic, 1>& add_grad);
        void resetGrad();
        void setOnesGrad();
        void addChild(const Tensor<T>& child);

        // graph computations
        void backward();
        void forward();

        // values
        void setValues(Eigen::Matrix<T, Eigen::Dynamic, 1> new_values);
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& getValues() const;
        void fill(T value);
        T& operator[](const std::vector<int>& multi_dim_index);

        // size etc...
        int size() const;
        std::vector<int> shape() const;
        void reshape(std::vector<int> dim);

        // for user
        void display() const;
        void displayGrad() const;
    };

    // name the float tensor
    using FTensor = Tensor<float>;
} // namespace name
