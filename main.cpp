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
    auto t1 = tensor::FTensor::random({2, 2});
    auto t2 = tensor::FTensor::random({1, 2});
    auto t3 = nn::math::clip(t1 - t2, 0.25f, 1.0f);
    // auto n = neuron::Neuron<float>(3);
    // auto output = n.forward(input);

    auto graph = graph::FComputeGraph(t3);
    graph.backward();

    std::cout << " T3 : " << std::endl;
    t3->display();
    t3->displayGrad();
    std::cout << " T2 : " << std::endl;
    t2->display();
    t2->displayGrad();
    std::cout << " T1 : " << std::endl;
    t1->display();
    t1->displayGrad();

    // auto t1 = tensor::FTensor::random({5, 3});
    // auto t2 = tensor::FTensor::random({1, 3});
    // auto t3 = tensor::FTensor::random({ 1 });
    // auto output = nn::math::reduceSum(t1 * t2, {1}) + t3;

    // auto graph = graph::FComputeGraph(output);
    // graph.backward();

    // output->display();
    return 0;
}