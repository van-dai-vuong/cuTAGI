#include "../../include/bindings/tagi_autograd_bindings.h"

#include <pybind11/stl.h>

#include "../../include/tagi_autograd.h"

namespace py = pybind11;
using namespace tagi_autograd;

void bind_tagi_autograd(py::module_& parent)
/* Exposes tagi_autograd under cutagi.tagi_autograd, a submodule of its own
 * because the top-level `cutagi` module already has unrelated `Linear`/
 * `Sequential`/`Parameter` classes for the production, hand-written-
 * backward layers.
 */
{
    py::module_ m = parent.def_submodule(
        "tagi_autograd",
        "Autograd for Bayesian inference (TAGI): define only the forward "
        "pass -- observe() runs the backward pass automatically.");

    m.def("set_trace", &set_trace, py::arg("enabled") = true);

    py::class_<GaussianTensor, std::shared_ptr<GaussianTensor>>(
        m, "GaussianTensor")
        .def_readonly("mu", &GaussianTensor::mu)
        .def_readonly("var", &GaussianTensor::var)
        .def_readonly("shape", &GaussianTensor::shape)
        .def_readwrite("name", &GaussianTensor::name)
        .def_readonly("post_mu", &GaussianTensor::post_mu)
        .def_readonly("post_var", &GaussianTensor::post_var)
        .def("named",
             [](GaussianTensor& self, const std::string& n) { self.named(n); })
        .def("retain", [](GaussianTensor& self) { self.retain(); })
        .def("detach", &GaussianTensor::detach)
        .def("observe", &GaussianTensor::observe, py::arg("y"),
             py::arg("var_v") = 0.0f)
        .def("backward", &GaussianTensor::backward)
        .def("__repr__", [](const GaussianTensor& t) {
            return "<GaussianTensor '" + t.name + "' shape=(" +
                   std::to_string(t.shape[0]) + "," +
                   std::to_string(t.shape[1]) + ")>";
        });

    py::class_<Parameter, GaussianTensor, std::shared_ptr<Parameter>>(
        m, "Parameter")
        .def(py::init<std::vector<float>, std::vector<float>, std::vector<int>,
                      std::string>(),
             py::arg("mu"), py::arg("var"), py::arg("shape"),
             py::arg("name") = "param");

    m.def("tensor", &tensor, py::arg("mu"), py::arg("shape"),
          py::arg("var") = std::vector<float>{}, py::arg("name") = "x");

    m.def("add", &add, py::arg("x"), py::arg("y"));
    m.def("mul", &mul, py::arg("x"), py::arg("y"));
    m.def("relu", &relu, py::arg("x"));
    m.def("tanh", &tanh_act, py::arg("x"));
    m.def("sigmoid", &sigmoid, py::arg("x"));
    m.def("mixture_relu", &mixture_relu, py::arg("x"));
    m.def("mixture_sigmoid", &mixture_sigmoid, py::arg("x"));
    m.def("mixture_tanh", &mixture_tanh, py::arg("x"));
    m.def("softplus", &softplus, py::arg("x"));
    m.def("leaky_relu", &leaky_relu, py::arg("x"), py::arg("alpha") = 0.1f);
    m.def("even_exp", &even_exp, py::arg("x"));
    m.def("chunk", &chunk, py::arg("x"), py::arg("chunks"),
          py::arg("name") = "");
    m.def("linear", &linear, py::arg("x"), py::arg("w"),
          py::arg("b") = nullptr);

    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("parameters", &Module::parameters)
        .def("__call__", &Module::operator());

    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int, bool, float, float, std::string, int>(),
             py::arg("n_in"), py::arg("n_out"), py::arg("bias") = true,
             py::arg("gain_w") = 1.0f, py::arg("gain_b") = 1.0f,
             py::arg("init_method") = "He", py::arg("fan_in_override") = 0)
        .def("forward", &Linear::forward)
        .def_readonly("weight", &Linear::weight)
        .def_readonly("bias", &Linear::bias);

    py::class_<Sequential, Module, std::shared_ptr<Sequential>>(m, "Sequential")
        .def(py::init<std::vector<std::shared_ptr<Module>>>(), py::arg("items"))
        .def("forward", &Sequential::forward);
}
