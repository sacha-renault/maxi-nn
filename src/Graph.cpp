#include "../include/MaxiNn.hpp"

namespace nn::graph
{
    template<typename T>
    ComputationGraph<T>::ComputationGraph(std::shared_ptr<tensor::Tensor<T>> root) {
        std::vector<std::shared_ptr<tensor::Tensor<T>>> sorted;
        std::unordered_set<std::shared_ptr<tensor::Tensor<T>>> visited;

        // create a lambda to recursively get all children nodes
        std::function<void(std::shared_ptr<tensor::Tensor<T>>)> dfs = [&](std::shared_ptr<tensor::Tensor<T>> node) {
            if (visited.count(node)) return;
            visited.insert(node);
            for (const auto& child : node->getChildren()) {
                dfs(child);
            }
            sorted.push_back(node);
        };

        // compute the actual graph
        dfs(root);

        // move ownership to this class
        nodes_ = std::move(sorted);
    }

    template<typename T>
    void ComputationGraph<T>::zeroGrad() {
        for(auto& node : nodes_) { // iterate over all the nodes
            node->resetGrad();
        }
    }

    template<typename T>
    void ComputationGraph<T>::forward() {
        for(auto& node : nodes_) { // iterate over all the nodes
            node->forward();
        }
    }

    template<typename T>
    void ComputationGraph<T>::backward() {
        int last_node_index = nodes_.size() - 1;
        nodes_[last_node_index]->setOnesGrad();
        for(int i = last_node_index ; i >= 0 ; --i) { // iterate over all the nodes
            auto node = nodes_[i];
            node->backward();
        }
    }

    template<typename T>
    const std::vector<std::shared_ptr<tensor::Tensor<T>>>& ComputationGraph<T>::getNodes() {
        return nodes_;
    }

    // explicit instanciation of the class
    template class ComputationGraph<float>;
    template class ComputationGraph<double>;
} // namespace nn::graph
