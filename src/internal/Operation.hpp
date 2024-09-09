#pragma once
#include <functional>
#include <eigen3/Eigen/Dense>
#include <vector>

namespace nn::Operation
{
    template <typename T>
    using BackwardFunc = std::function<std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>(
        std::vector<Eigen::Matrix<T, Eigen::Dynamic, 1>>&,  // all the values of children
        Eigen::Matrix<T, Eigen::Dynamic, 1>&,               // values of parent
        Eigen::Matrix<T, Eigen::Dynamic, 1>&                // grad of parent
    )>;
} // namespace nn::Operation
