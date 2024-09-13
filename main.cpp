#include "include/MaxiNn.hpp"
#include "src/InternalOperation/InternalOperation.hpp"
#include <iostream>

using namespace nn;

int main() {
    ulong bs = 12;
    ulong outputsize = 5;
    int num_epoch = 100;
    float lr = 1e-2;

    auto input = tensor::FTensor::normal({bs, 256});

    auto layer1 = layers::FFcc(256, 128, nn::math::tanh<float>);
    auto layer2 = layers::FFcc(128, 64, nn::math::tanh<float>);
    auto layer3 = layers::FFcc(64, 32, nn::math::tanh<float>);
    auto layer4 = layers::FFcc(32, outputsize, nn::math::softmax<float>);

    auto layers_vec = {layer1, layer2, layer3, layer4};

    auto y_true = tensor::FTensor::zeros({bs, outputsize});

    for (size_t i = 0 ; i < bs ; ++i) {
        y_true->getItem({i, i % outputsize}) = 1;
    }

    auto x = layer1(input);
    x = layer2(x);
    x = layer3(x);
    x = layer4(x);

    auto batch_loss = nn::loss::categoricalCrossEntropy(x, y_true);

    auto graph = graph::FComputationGraph(batch_loss);

    for (int i = 0 ; i < num_epoch ; ++i) {
        auto rnd = tensor::FTensor::normal(input->shape());
        graph.forward();

        // std::cout << x->getValues() << std::endl;

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