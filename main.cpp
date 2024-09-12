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
    ulong bs = 128;
    ulong outputsize = 1;
    int num_epoch = 500;
    float lr = 1e-2;

    auto input = tensor::FTensor::random({bs, 256}, -1, 1);

    float l1 = std::sqrt(6.0 / (256 + 128));

    auto layer1 = layers::FFcc(256, 128, nn::math::tanh<float>);
    auto layer2 = layers::FFcc(128, 64, nn::math::tanh<float>);
    auto layer3 = layers::FFcc(64, 32, nn::math::tanh<float>);
    auto layer4 = layers::FFcc(32, outputsize, nn::math::tanh<float>);

    auto layers_vec = {layer1, layer2, layer3, layer4};

    auto y_true = tensor::FTensor::random({bs, outputsize}, -1, 1);

    auto x = layer1(input);
    x = layer2(x);
    x = layer3(x);
    x = layer4(x);

    auto batch_loss = nn::loss::meanSquaredError(x, y_true);

    auto graph = graph::FComputationGraph(batch_loss);

    for (int i = 0 ; i < num_epoch ; ++i) {
        auto rnd = tensor::FTensor::normal(input->shape());
        graph.forward();
        graph.zeroGrad();
        graph.backward();


        for (auto node : layers_vec) {
            auto w = node.getWeights();
            auto gradsw = w->getGrad();
            auto valuesw = w->getValues();
            w->setValues(valuesw - gradsw*lr);

            auto b = node.getWeights();
            auto gradsb = b->getGrad();
            auto valuesb = b->getValues();
            b->setValues(valuesb - gradsb*lr);
        }

        std::cout << " iteration : " << i + 1 << " ; Loss : " << batch_loss->getItem({0}) << " ; LR : " << lr << std::endl;

        if ((i + 1) % 50 == 0) {
            lr *= 0.9;
        }
    }

    return 0;
}

// int main() {
//     ulong bs = 1;
//     ulong outputsize = 1;
//     float lr = 1e-3;

//     auto input = tensor::FTensor::normal({bs, 4});

//     auto layer1 = tensor::FTensor::normal({4, 2});
//     auto bias1 = tensor::FTensor::normal({1, 2});

//     auto layer2 = tensor::FTensor::normal({2, 1});
//     auto bias2 = tensor::FTensor::normal({1, 1});

//     auto weights = {layer1, bias1, layer2, bias2};

//     auto y_true = tensor::FTensor::normal({bs, outputsize});

//     auto x = nn::math::dot(input, layer1);
//     x = x + bias1;
//     x = nn::math::tanh(x);
//     x = nn::math::dot(x, layer2);
//     x = x + bias2;
//     x = nn::math::tanh(x);
//     auto batch_loss = nn::loss::meanSquaredError(x, y_true);

//     auto graph = graph::FComputeGraph(batch_loss);

//     for (int i = 0 ; i < 100 ; ++i) {
//         auto rnd = tensor::FTensor::normal(input->shape());
//         input->fill(rnd->getValues());
//         graph.forward();
//         graph.zeroGrad();
//         graph.backward();



//         for (auto node : weights) {
//             auto grads = node->getGrad();
//             auto values = node->getValues();
//             node->setValues(values - grads*lr);
//         }

//         std::cout << " Real : " << y_true->getItem({0}) <<std::endl;
//         std::cout << " Output : " << x->getItem({0}) <<std::endl;
//         std::cout << " iteration : " << i + 1 << " ; Loss : " << batch_loss->getItem({0}) <<std::endl;
//         std::cout << " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" <<std::endl;
//         // std::cout << "  Real : " << node->  << batch_loss->getItem({0}) <<std::endl;
//     }

//     // input->displayGrad();
//     return 0;
// }

// int main() {
//     auto y1 = tensor::FTensor::ones({128, 784}) * 0.5f;
//     auto y2 = tensor::FTensor::zeros({128, 784});

//     auto result = loss::meanAbsoluteError(y1, y2);

//     result->display();
//     return 0;
// }