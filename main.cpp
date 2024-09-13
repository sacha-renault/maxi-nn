#include "include/MaxiNn.hpp"
#include "src/InternalOperation/InternalOperation.hpp"
#include <iostream>

using namespace nn;

int main() {
    ulong bs = 16;
    ulong b_number = 128; 
    ulong outputsize = 5;
    int num_epoch = 100;
    float lr = 1e-2;

    auto data = tensor::FTensor::normal({bs * b_number, 256});
    auto ground_trust = tensor::FTensor::zeros({bs * b_number, outputsize});

    auto input = tensor::FTensor::zeros({bs, 256});
    auto layer1 = layers::FFcc(256, 128, nn::math::tanh<float>);
    auto layer2 = layers::FFcc(128, 64, nn::math::tanh<float>);
    auto layer3 = layers::FFcc(64, 32, nn::math::tanh<float>);
    auto layer4 = layers::FFcc(32, outputsize, nn::math::softmax<float>);

    auto layers_vec = {layer1, layer2, layer3, layer4};

    auto y_true = tensor::FTensor::zeros({bs, outputsize});

    for (size_t i = 0 ; i < bs * b_number; ++i) {
        ground_trust->getItem({i, i % outputsize}) = 1;
    }

    auto x = layer1(input);
    x = layer2(x);
    x = layer3(x);
    x = layer4(x);
    auto batch_loss = nn::loss::categoricalCrossEntropy(x, y_true);
    auto graph = graph::FComputationGraph(batch_loss);

    for (int epoch = 1 ; epoch < num_epoch + 1 ; ++epoch) {
        // init a global loss
        float epoch_loss = 0;

        for (int current_batch = 0 ; current_batch < b_number ; ++current_batch) {
            // Set the data in x and y
            auto batch_data = data->slice({ strides(current_batch * bs, (current_batch + 1) * bs) });
            auto batch_gt = ground_trust->slice({ strides(current_batch * bs, (current_batch + 1) * bs) });
            y_true->fill(batch_gt->getValues());
            input->fill(batch_data->getValues());

            // compute graph
            graph.zeroGrad();
            graph.forward();
            graph.backward();

            // update weights
            for (auto node : layers_vec) {
                auto w = node.getWeights();
                auto gradsw = w->getGrad();
                auto valuesw = w->getValues();
                w->setValues(valuesw - gradsw*lr);
                auto bias = node.getWeights();
                auto gradsb = bias->getGrad();
                auto valuesb = bias->getValues();
                bias->setValues(valuesb - gradsb*lr);
            }

            // increment loss
            epoch_loss += batch_loss->getItem({0});

            std::cout << " iteration : " << epoch << " ; Batch : " << current_batch + 1 << " / " << b_number;
            std::cout << " ; Loss : " << epoch_loss / (current_batch + 1) << " ; LR : " << lr << "\r" << std::flush;
        }

        std::cout << std::endl;

        if ((epoch + 1) % 50 == 0) {
            lr *= 0.9;
        }
    }

    return 0;
}
