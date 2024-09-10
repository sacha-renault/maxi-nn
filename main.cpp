#include "include/MaxiNn.hpp"
#include "src/InternalOperation/InternalOperation.hpp"
#include <iostream>

using namespace nn;

int main() {
    // auto t1 = tensor::FTensor::random({2, 2});
    // auto t2 = tensor::FTensor::random({2, 2});
    // auto t3 = tensor::FTensor::normal({2, 2});
    // (*t1)[{0,0}] = 0;

    // auto y = FTensor::create({2,2}, true);
    // y->fill(-1);
    // (*y).getItem({0,0}) = 1;

    // auto t5 = math::tanh(t1 * t2 - t3);
    // auto t7 = loss::meanSquaredError(t5, y);
    // // auto t5 = t1 * t2 - t3;

    // auto graph = graph::FComputeGraph(t7);
    // graph.backward();

    // t7->display();
    // y->display();

    // t5->displayGrad();
    // t3->displayGrad();
    // t2->displayGrad();
    // t1->displayGrad();

    Eigen::Matrix<float, -1, 1> x(1);
    x << 1.0;

    std::vector<int> current_shape = {1, 1};
    std::vector<int> target_shape = {3, 1};

    Eigen::Matrix<float, -1, 1> y = Operation::broadcastView<float>(x, current_shape, target_shape);

    // Modify the broadcasted view
    x.array() += 1;

    // The original matrix x will be updated
    std::cout << "Updated x:\n" << x << std::endl;  // Outputs: 4 (since it's broadcasted 3 times and added 1 each time)
    std::cout << "y :\n" << y << std::endl;  // Outputs: 4 (since it's broadcasted 3 times and added 1 each time)

    return 0;
}