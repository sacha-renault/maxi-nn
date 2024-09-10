#include "include/MaxiNn.hpp"
#include "src/InternalOperation/InternalOperation.hpp"
#include <iostream>

using namespace nn;

int main() {
    auto t1 = tensor::FTensor::create({2, 2}, true);
    auto t2 = tensor::FTensor::create({2, 2}, true);
    auto t3 = tensor::FTensor::create({2, 2}, true);
    t1->fill(1);
    t2->fill(2);
    t3->fill(3);
    (*t1)[{0,0}] = 5;

    auto y = FTensor::create({2,2}, true);
    y->fill(-1);
    (*y)[{0,0}] = 1;

    auto t5 = math::tanh(t1 * t2 - t3);
    auto t7 = loss::meanSquaredError(t5, y);
    // auto t5 = t1 * t2 - t3;

    auto graph = graph::FComputeGraph(t7);
    graph.backward();

    t7->display();

    t5->displayGrad();
    t3->displayGrad();
    t2->displayGrad();
    t1->displayGrad();

    return 0;
}