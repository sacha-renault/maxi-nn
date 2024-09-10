#include "include/MaxiNn.hpp"
#include "src/InternalOperation/InternalOperation.hpp"
#include <iostream>

int main() {
    nn::tensor::FTensor t1({2, 2}, true);
    nn::tensor::FTensor t2({2, 2}, true);
    nn::tensor::FTensor t3({2, 2}, true);
    t1.fill(1);
    t2.fill(2);
    t3.fill(3);
    t1[{0,0}] = 5;

    auto t4 = (t1 - t2);
    auto t5 = t4 * t3;

    t5.setOnesGrad();
    t5.backward();

    t5.display();

    return 0;
}