#include "include/MaxiNn.hpp"
#include "src/internal/Operator.hpp"
#include <iostream>

int main() {
    auto t1 = nn::tensor::FTensor({2, 2}, true);
    auto t2 = nn::tensor::FTensor({2, 2}, true);
    t1.fill(1);
    t2.fill(2);
    t1[{0,0}] = 5;

    auto t3 = t1 * t2;
    
    t3.setOnesGrad();
    t3.backward();

    return 0;
}