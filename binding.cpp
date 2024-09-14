#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <xtensor-python/pytensor.hpp>
#include <xtensor-python/pyarray.hpp>

#include "include/MaxiNn.hpp"  // Change this include to the library you are binding

namespace py = pybind11;

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  AUTOGENERATED CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// <litgen_glue_code>  // Autogenerated code below! Do not edit!

// </litgen_glue_code> // Autogenerated code end
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  AUTOGENERATED CODE END !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


// You can add any code here


PYBIND11_MODULE(nn, m)      //  rename this function name!!!
{
    // You can add any code here


    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  AUTOGENERATED CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // <litgen_pydef> // Autogenerated code below! Do not edit!
    ////////////////////    <generated_from:Tensor.hpp>    ////////////////////

    { // <namespace tensor>
        py::class_<nn::Operation::IOperation<float>>(m, "Operation_float");
        py::class_<xt::svector<unsigned long, 4ul, std::allocator<unsigned long>, true>>(m, "SVectorULong")
            .def(py::init<>())
            .def(py::init<std::vector<unsigned long>>())
            .def("size", &xt::svector<unsigned long, 4ul, std::allocator<unsigned long>, true>::size)
            .def("__getitem__", [](const xt::svector<unsigned long, 4ul, std::allocator<unsigned long>, true>& v, size_t i) {
                return v[i];
            })
            .def("__setitem__", [](xt::svector<unsigned long, 4ul, std::allocator<unsigned long>, true>& v, size_t i, unsigned long val) {
                v[i] = val;
            });

        auto tensorNsp = m.def_submodule("tensor", "namespace name");
        py::enum_<nn::tensor::TensorType>(tensorNsp, "TensorType", py::arithmetic(), "")
            .value("input", nn::tensor::Input, "")
            .value("output", nn::tensor::Output, "")
            .value("parameter", nn::tensor::Parameter, "")
            .value("none", nn::tensor::None, "");


            py::class_<nn::tensor::Tensor<float>, std::shared_ptr<nn::tensor::Tensor<float>>>
                (tensorNsp, "Tensor_float", "")
            .def_static("create",
                [](nn::tensor::Tensor<float> & self) { return self.create(); })
            .def_static("create",
                py::overload_cast<xt::dynamic_shape<size_t>, bool>(&nn::tensor::Tensor<float>::create), py::arg("dim"), py::arg("requires_grad") = true)
            .def_static("create",
                py::overload_cast<xt::xarray<float>, bool>(&nn::tensor::Tensor<float>::create), py::arg("values"), py::arg("requires_grad") = true)
            .def_static("create",
                py::overload_cast<xt::xarray<float>, std::shared_ptr<nn::Operation::IOperation<float>>, bool>(&nn::tensor::Tensor<float>::create), py::arg("values"), py::arg("stream"), py::arg("requires_grad") = true)
            .def_static("zeros",
                &nn::tensor::Tensor<float>::zeros, py::arg("dim"), py::arg("requires_grad") = true)
            .def_static("ones",
                &nn::tensor::Tensor<float>::ones, py::arg("dim"), py::arg("requires_grad") = true)
            .def_static("random",
                &nn::tensor::Tensor<float>::random, py::arg("dim"), py::arg("min") = 0, py::arg("max") = 1, py::arg("requires_grad") = true)
            .def_static("normal",
                &nn::tensor::Tensor<float>::normal, py::arg("dim"), py::arg("mean") = 0, py::arg("stddev") = 1, py::arg("requires_grad") = true)
            .def("set_tensor_type",
                &nn::tensor::Tensor<float>::setTensorType,
                py::arg("type"),
                "set a type for the tensor")
            .def("accumulate_grad",
                &nn::tensor::Tensor<float>::accumulateGrad, py::arg("add_grad"))
            .def("set_grad",
                &nn::tensor::Tensor<float>::setGrad, py::arg("grads"))
            .def("get_grad",
                &nn::tensor::Tensor<float>::getGrad)
            .def("reset_grad",
                &nn::tensor::Tensor<float>::resetGrad)
            .def("set_ones_grad",
                &nn::tensor::Tensor<float>::setOnesGrad)
            .def("add_child",
                &nn::tensor::Tensor<float>::addChild, py::arg("child"))
            .def("get_children",
                &nn::tensor::Tensor<float>::getChildren)
            .def("backward",
                &nn::tensor::Tensor<float>::backward)
            .def("forward",
                &nn::tensor::Tensor<float>::forward)
            .def("set_values",
                &nn::tensor::Tensor<float>::setValues, py::arg("new_values"))
            .def("get_values",
                &nn::tensor::Tensor<float>::getValues)
            .def("fill",
                py::overload_cast<float>(&nn::tensor::Tensor<float>::fill), py::arg("value"))
            .def("fill",
                py::overload_cast<xt::xarray<float>>(&nn::tensor::Tensor<float>::fill), py::arg("values"))
            .def("__getitem__",
                &nn::tensor::Tensor<float>::operator[], py::arg("idx"))
            .def("get_item",
                &nn::tensor::Tensor<float>::getItem, py::arg("idx"))
            .def("slice",
                &nn::tensor::Tensor<float>::slice, py::arg("slices"))
            .def("size",
                &nn::tensor::Tensor<float>::size)
            .def("shape",
                &nn::tensor::Tensor<float>::shape)
            .def("display",
                &nn::tensor::Tensor<float>::display)
            .def("display_grad",
                &nn::tensor::Tensor<float>::displayGrad)
            ;
    } // </namespace tensor>
    ////////////////////    </generated_from:Tensor.hpp>    ////////////////////


    ////////////////////    <generated_from:LayerFcc.hpp>    ////////////////////

    { // <namespace layers>
        auto layerNsp = m.def_submodule("layers", "namespace nn::layers");
            py::class_<nn::layers::Fcc<float>>
                (layerNsp, "Fcc_float", "")
            .def(py::init<size_t, size_t>(),
                py::arg("num_input"), py::arg("num_output"))
            .def(py::init<size_t, size_t, std::function<std::shared_ptr<nn::tensor::Tensor<float>>(std::shared_ptr<nn::tensor::Tensor<float>>)>>(),
                py::arg("num_input"), py::arg("num_output"), py::arg("activation"))
            .def("__call__",
                &nn::layers::Fcc<float>::operator(), py::arg("input"))
            .def("get_weights",
                &nn::layers::Fcc<float>::getWeights)
            .def("get_bias",
                &nn::layers::Fcc<float>::getBias)
            ;
    } // </namespace layers>
    ////////////////////    </generated_from:LayerFcc.hpp>    ////////////////////


    ////////////////////    <generated_from:TensorwiseMath.hpp>    ////////////////////

    { // <namespace math>
        auto mathNsp = m.def_submodule("math", "namespace nn::math");
        mathNsp.def("dot",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>, std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::dot<float>), py::arg("lt"), py::arg("rt"));

        mathNsp.def("reduce_sum",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>, xt::dynamic_shape<size_t>>(nn::math::reduceSum<float>), py::arg("input"), py::arg("axis") = xt::dynamic_shape<size_t>());

        mathNsp.def("reduce_mean",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>, xt::dynamic_shape<size_t>>(nn::math::reduceMean<float>), py::arg("input"), py::arg("axis") = xt::dynamic_shape<size_t>());

        mathNsp.def("softmax",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::softmax<float>),
            py::arg("input"),
            "/ @brief softmax funtion on dimension 1 (expect tensor of shape (batch_size, num_inputs))\n/ @param input tensor\n/ @return softmaxed tensor");
    } // </mathNsp math>
    ////////////////////    </generated_from:TensorwiseMath.hpp>    ////////////////////


    ////////////////////    <generated_from:ElementwiseMath.hpp>    ////////////////////

    { // <mathNsp math>
        auto mathNsp2 = m.def_submodule("math", "namespace nn::math");
        mathNsp2.def("relu",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::relu<float>), py::arg("input"));

        mathNsp2.def("tanh",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::tanh<float>), py::arg("input"));

        mathNsp2.def("pow",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>, float>(nn::math::pow<float>), py::arg("input"), py::arg("exponent"));

        mathNsp2.def("abs",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::abs<float>), py::arg("input"));

        mathNsp2.def("exp",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::exp<float>), py::arg("input"));

        mathNsp2.def("log",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::log<float>), py::arg("input"));

        mathNsp2.def("sqrt",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>>(nn::math::sqrt<float>), py::arg("input"));

        mathNsp2.def("clip",
            py::overload_cast<std::shared_ptr<nn::tensor::Tensor<float>>, float, float>(nn::math::clip<float>), py::arg("input"), py::arg("min"), py::arg("max"));
    } // </namespace math>
    ////////////////////    </generated_from:ElementwiseMath.hpp>    ////////////////////


    ////////////////////    <generated_from:Loss.hpp>    ////////////////////

    { // <namespace loss>
        auto loss_float = m.def_submodule("loss", "namespace nn::Loss");
        loss_float.def("mean_squared_error",
            py::overload_cast<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>>(nn::loss::meanSquaredError<float>), py::arg("pred"), py::arg("real"));

        loss_float.def("mean_absolute_error",
            py::overload_cast<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>>(nn::loss::meanAbsoluteError<float>), py::arg("pred"), py::arg("real"));

        loss_float.def("categorical_cross_entropy",
            py::overload_cast<std::shared_ptr<Tensor<float>>, std::shared_ptr<Tensor<float>>, float>(nn::loss::categoricalCrossEntropy<float>), py::arg("pred"), py::arg("real"), py::arg("epsilon") = 1e-10);
    } // </namespace loss>
    ////////////////////    </generated_from:Loss.hpp>    ////////////////////
    { // <namespace graph>
        auto graphNsp = m.def_submodule("graph", "namespace nn::gradient");
            py::class_<nn::graph::ComputationGraph<float>>
                (graphNsp, "ComputationGraph_float", "")
            .def(py::init<std::shared_ptr<nn::tensor::Tensor<float>>>(),
                py::arg("root"))
            .def("get_nodes",
                &nn::graph::ComputationGraph<float>::getNodes, "/ @brief get nodes in one topological sort\n/ @return sortes nodes of the graph")
            .def("forward",
                &nn::graph::ComputationGraph<float>::forward, "/ @brief Apply forward() on each node of the graph")
            .def("backward",
                &nn::graph::ComputationGraph<float>::backward, "/ @brief Apply backward() on each node of the graph (reverse order)")
            .def("zero_grad",
                &nn::graph::ComputationGraph<float>::zeroGrad, "/ @brief Set gradient to 0 to every node")
            ;
    } // </namespace graph>

    // </litgen_pydef> // Autogenerated code end
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  AUTOGENERATED CODE END !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}