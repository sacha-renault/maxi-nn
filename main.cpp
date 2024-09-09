#include "include/MaxiNn.hpp"
#include "src/internal/Operator.hpp"
#include <iostream>

int main() {
    auto t1 = nn::tensor::FTensor({2, 2}, true);
    auto t2 = nn::tensor::FTensor({2, 2}, true);
    t1.fill(1);
    t2.fill(2);

    auto t3 = t1 * t2;
    
    for (int i = 0 ; i < 2 ; ++i){
        std::cout << t3[{i,i}] << std::endl;
    }
    
    return 0;
}