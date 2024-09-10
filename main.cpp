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

    auto t5 = t1 * t2 - t3;

    auto graph = graph::FComputeGraph(t5);
    graph.backward();

    t5->display();
    t1->display();

    t5->displayGrad();
    t3->displayGrad();
    t2->displayGrad();
    t1->displayGrad();

    return 0;
}