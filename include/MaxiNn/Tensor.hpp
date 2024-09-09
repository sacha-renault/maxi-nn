#pragma once
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>

namespace nn::tensor
{

    template <typename T>
    class Tensor{
    private:
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

        // private functions
        int computeIndex(const std::vector<int>& multi_dim_index) const;
    public:
        ~Tensor() = default;
        Tensor();
        Tensor(std::vector<int> dim, bool requires_grad = true);
        Tensor(std::vector<int> dim, Eigen::Matrix<T, Eigen::Dynamic, 1> values, bool requires_grad = true);

        // gradient
        void accumulateGrad(const Eigen::Matrix<T, Eigen::Dynamic, 1>& add_grad);
        void resetGrad();
        void addChild(const Tensor<T>& child);
        void backward();

        // values
        void setValues(Eigen::Matrix<T, Eigen::Dynamic, 1> new_values);
        const Eigen::Matrix<T, Eigen::Dynamic, 1>& getValues() const;
        void fill(T value);
        T& operator[](const std::vector<int>& multi_dim_index);

        // size etc...
        int size() const;
        std::vector<int> shape() const;
        void reshape(std::vector<int> dim);
    };

    // name the float tensor
    using FTensor = Tensor<float>;
} // namespace name
