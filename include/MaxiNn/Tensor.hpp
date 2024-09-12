#pragma once
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>
#include <memory>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include "../../src/InternalOperation/InternalOperation.hpp"

namespace nn::tensor
{
    template <typename T>
    class Tensor{
    protected:
        // values and grads
        xt::xarray<T> values_;
        xt::xarray<T> grads_;

        // bools
        bool requires_grad_;

        // dimensions
        xt::dynamic_shape<size_t> dimensions_;
        int total_size_;

        // children
        std::vector<std::shared_ptr<Tensor>> children_;

        // backward function
        std::shared_ptr<nn::Operation::IOperation<T>> stream_ptr;

        // all constructor must be PRIVATE (we only want to use shared ptr)
        Tensor();
        Tensor(xt::dynamic_shape<size_t> dim, bool requires_grad = true);
        Tensor(xt::dynamic_shape<size_t> dim, xt::xarray<T> values, bool requires_grad = true);
        Tensor(xt::dynamic_shape<size_t> dim, xt::xarray<T> values, std::shared_ptr<nn::Operation::IOperation<T>> stream, bool requires_grad = true);
    public:
        // static factory method
        static std::shared_ptr<Tensor<T>> create();
        static std::shared_ptr<Tensor<T>> create(xt::dynamic_shape<size_t> dim, bool requires_grad = true);
        static std::shared_ptr<Tensor<T>> create(xt::dynamic_shape<size_t> dim, xt::xarray<T> values, bool requires_grad = true);
        static std::shared_ptr<Tensor<T>> create(xt::dynamic_shape<size_t> dim, xt::xarray<T> values, std::shared_ptr<nn::Operation::IOperation<T>> stream, bool requires_grad = true);
        static std::shared_ptr<Tensor<T>> zeros(xt::dynamic_shape<size_t> dim, bool requires_grad = true);
        static std::shared_ptr<Tensor<T>> ones(xt::dynamic_shape<size_t> dim, bool requires_grad = true);
        static std::shared_ptr<Tensor<T>> random(xt::dynamic_shape<size_t> dim, T min = 0, T max = 1, bool requires_grad = true);
        static std::shared_ptr<Tensor<T>> normal(xt::dynamic_shape<size_t> dim, T mean = 0, T stddev = 1, bool requires_grad = true);

        // gradient
        void accumulateGrad(const xt::xarray<T>& add_grad);
        void setGrad(xt::xarray<T> grads);
        const xt::xarray<T>& getGrad() const;
        void resetGrad();
        void setOnesGrad();
        void addChild(std::shared_ptr<Tensor<T>> child);
        const std::vector<std::shared_ptr<Tensor<T>>>& getChildren() const;

        // graph computations
        void backward();
        void forward();

        // values
        void setValues(xt::xarray<T> new_values);
        const xt::xarray<T>& getValues() const;
        void fill(T value);
        void fill(xt::xarray<T> values);
        const xt::xarray<T>& getValues();
        T& operator[](const xt::xindex& idx);
        T& getItem(const xt::xindex& idx);

        // size etc...
        int size() const;
        xt::dynamic_shape<size_t> shape() const;
        void reshape(xt::dynamic_shape<size_t> dim);

        // for user
        void display() const;
        void displayGrad() const;
    };

    // name the float tensor
    using FTensor = Tensor<float>;
} // namespace name
