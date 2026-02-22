#include "../include/bindings/tlstm_layer_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "../include/tlstm_layer.h"

void bind_tlstm_layer(pybind11::module_& modo) {
    pybind11::class_<TLSTM, std::shared_ptr<TLSTM>, BaseLayer>(modo, "TLSTM")
        .def(pybind11::init<size_t, size_t, int, bool, float, float,
                            std::string, int>(),
             pybind11::arg("input_size"), pybind11::arg("output_size"),
             pybind11::arg("seq_len"), pybind11::arg("bias"),
             pybind11::arg("gain_weight") = 1.0f,
             pybind11::arg("gain_bias") = 1.0f, pybind11::arg("method") = "He",
             pybind11::arg("device_idx") = 0)
        .def("get_layer_info", &TLSTM::get_layer_info)
        .def("get_layer_name", &TLSTM::get_layer_name)
        .def_readwrite("gain_w", &TLSTM::gain_w)
        .def_readwrite("gain_b", &TLSTM::gain_b)
        .def_readwrite("init_method", &TLSTM::init_method)
        .def("init_weight_bias", &TLSTM::init_weight_bias)
        .def("forward", &TLSTM::forward)
        .def("backward", &TLSTM::backward);
}
