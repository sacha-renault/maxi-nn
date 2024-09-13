#pragma once
#include <unordered_set>
#include <functional>
#include "Tensor.hpp"

namespace nn::graph
{
    template<typename T>
    class ComputationGraph {
    protected:
        std::vector<std::shared_ptr<tensor::Tensor<T>>> nodes_;

    public:
        ComputationGraph(std::shared_ptr<tensor::Tensor<T>> root);

        /// @brief get nodes in one topological sort
        /// @return sortes nodes of the graph
        const std::vector<std::shared_ptr<tensor::Tensor<T>>>& getNodes();

        /// @brief Apply forward() on each node of the graph
        void forward();

        /// @brief Apply backward() on each node of the graph (reverse order)
        void backward();

        /// @brief Set gradient to 0 to every node
        void zeroGrad();
    };

    using FComputationGraph = ComputationGraph<float>;
} // namespace nn::gradient
