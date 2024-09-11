#include "include/MaxiNn.hpp"
#include "src/InternalOperation/InternalOperation.hpp"
#include <iostream>

using namespace nn;

// int main() {
//     auto t1 = tensor::FTensor::random({2, 2});
//     auto t2 = tensor::FTensor::random({2, 2});
//     auto t3 = tensor::FTensor::normal({2, 2});
//     (*t1)[{0,0}] = 0;

//     auto y = FTensor::create({2,2}, true);
//     y->fill(-1);
//     (*y).getItem({0,0}) = 1;

//     auto t5 = math::tanh(t1 * t2 - t3);
//     auto t7 = loss::meanSquaredError(t5, y);
//     // auto t5 = t1 * t2 - t3;

//     auto graph = graph::FComputeGraph(t7);
//     graph.backward();

//     t7->display();
//     y->display();
//     t5->display();

//     t5->displayGrad();
//     t3->displayGrad();
//     t2->displayGrad();
//     t1->displayGrad();

//     return 0;
// }

int main() {
    auto input = tensor::FTensor::normal({128, 784});
    auto t2 = tensor::FTensor::normal({784, 256});
    auto t3 = tensor::FTensor::normal({256, 128});
    auto t4 = tensor::FTensor::normal({128, 32});
    auto t5 = tensor::FTensor::normal({32, 10});

    auto x = nn::math::dot(input, t2);
    x = nn::math::tanh(x);
    x = nn::math::dot(x, t3);
    x = nn::math::tanh(x);
    x = nn::math::dot(x, t4);
    x = nn::math::tanh(x);
    x = nn::math::dot(x, t5);
    x = nn::math::tanh(x);
    // auto n = neuron::Neuron<float>(3);
    // auto output = n.forward(input);

    auto graph = graph::FComputeGraph(x);
    graph.backward();

    for (int i = 0 ; i < 100 ; ++i) {
        auto rnd = tensor::FTensor::normal(input->shape());
        input->fill(rnd->getValues());
        graph.forward();
        std::cout << " iteration : " << i << std::endl;
    }
    
    // x->displayGrad();

    // auto t1 = tensor::FTensor::random({5, 3});
    // auto t2 = tensor::FTensor::random({1, 3});
    // auto t3 = tensor::FTensor::random({ 1 });
    // auto output = nn::math::reduceSum(t1 * t2, {1}) + t3;

    // auto graph = graph::FComputeGraph(output);
    // graph.backward();

    // output->display();
    return 0;
}