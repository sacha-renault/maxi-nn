#pragma once
#include "Tensor.hpp"
#include <unordered_set>

namespace nn::graph
{
    template<typename T>
    class ComputeGraph {
    protected:
        std::vector<std::shared_ptr<tensor::Tensor<T>>> nodes_;

    public:
        ComputeGraph(std::shared_ptr<tensor::Tensor<T>> root);
        void forward();
        void backward();
        const std::vector<std::shared_ptr<tensor::Tensor<T>>>& getNodes();
        void zeroGrad();
    };

    using FComputeGraph = ComputeGraph<float>;
} // namespace nn::gradient
