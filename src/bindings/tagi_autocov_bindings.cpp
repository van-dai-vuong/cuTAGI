#include "../../include/bindings/tagi_autocov_bindings.h"

#include <pybind11/stl.h>

#include "../../include/tagi_autocov.h"

namespace py = pybind11;
using namespace tagi_autocov;

void bind_tagi_autocov(py::module_& parent)
/* Exposes tagi_autocov under cutagi.tagi_autocov, a submodule of its own
 * because the top-level `cutagi` module already has unrelated `Linear`/
 * `Sequential`/`Parameter` classes for the production, hand-written-
 * backward layers.
 */
{
    py::module_ m = parent.def_submodule(
        "tagi_autocov",
        "Autograd for Bayesian inference (TAGI): define only the forward "
        "pass -- observe() runs the backward pass automatically.");

    m.def("set_trace", &set_trace, py::arg("enabled") = true);
    m.def("clear_cov_tapes", &clear_cov_tapes,
          "Free every covariance tape; call at the start of a rollout step, "
          "before track_cov()-ing that step's roots.");
    m.def(
        "cross_cov",
        [](const TensorPtr& x, const TensorPtr& y) {
            // Reshape the flat per-batch-row blocks to nested lists:
            // (cols_x, cols_y) for single-row nodes, else
            // (rows, cols_x, cols_y). Cross-batch covariance is
            // structurally zero and not returned.
            std::vector<float> flat = cross_cov(x, y);
            py::list out;
            if (flat.empty()) return out;
            int rows = x->shape[0];
            int cx = x->shape[1], cy = y->shape[1];
            for (int b = 0; b < rows; ++b) {
                py::list block;
                for (int i = 0; i < cx; ++i) {
                    py::list row;
                    for (int j = 0; j < cy; ++j) {
                        row.append(
                            flat[(static_cast<size_t>(b) * cx + i) * cy + j]);
                    }
                    block.append(row);
                }
                if (rows == 1) return block;
                out.append(block);
            }
            return out;
        },
        py::arg("x"), py::arg("y"),
        "Cross-covariance blocks cov(x[b,i], y[b,j]) from the tape's "
        "shared roots: shape (cols_x, cols_y) for single-row nodes, else "
        "(rows, cols_x, cols_y) -- one block per batch row, since batch "
        "rows are independent. E.g. cov(x, linear(x)) for x {1,3} and "
        "output {1,5} is 3x5; for {16,8} gates it is (16, 8, 8). "
        "add()/mul() consume each block's diagonal internally. Empty "
        "list if x and y share no tracked roots.");

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
        .def(
            "track_cov", [](GaussianTensor& self) { self.track_cov(); },
            "Mark as a stochastic root of the covariance tape, so ops "
            "downstream track their cross-covariances w.r.t. this node.")
        .def(
            "set_moments",
            [](GaussianTensor& self, std::vector<float> mu,
               std::vector<float> var) {
                if (mu.size() != self.mu.size() ||
                    var.size() != self.var.size()) {
                    throw std::invalid_argument(
                        "set_moments: size mismatch with existing moments");
                }
                self.mu = std::move(mu);
                self.var = std::move(var);
            },
            py::arg("mu"), py::arg("var"),
            "Overwrite (mu, var) in place, e.g. to copy weights trained in "
            "another framework into a Parameter for comparison.")
        .def_property_readonly(
            "cov_tape",
            [](const GaussianTensor& self) {
                // Debug view of the covariance tape: one dict per root,
                // with the linearized Jacobian d(self)/d(root) reshaped
                // to (numel_self, root_cols) nested lists.
                py::list entries;
                int numel = static_cast<int>(self.mu.size());
                for (const auto& e : self.cov_tape) {
                    py::dict d;
                    d["root"] = e.root->name;
                    d["root_cols"] = e.root_cols;
                    py::list jac;
                    for (int i = 0; i < numel; ++i) {
                        py::list row;
                        for (int k = 0; k < e.root_cols; ++k) {
                            row.append(
                                e.jac[static_cast<size_t>(i) * e.root_cols +
                                      k]);
                        }
                        jac.append(row);
                    }
                    d["jac"] = jac;
                    entries.append(d);
                }
                return entries;
            },
            "Covariance-tape entries: [{'root', 'root_cols', 'jac'}] where "
            "jac[i][k] = d self[i] / d root[k] (linearized). Empty when "
            "nothing upstream was track_cov()-ed.")
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
    m.def("scale", &scale, py::arg("x"), py::arg("c"));
    m.def("gather", &gather, py::arg("x"), py::arg("indices"), py::arg("rows"),
          py::arg("cols"), py::arg("name") = "");
    m.def("matmul", &matmul, py::arg("a"), py::arg("b"), py::arg("batch"),
          py::arg("m"), py::arg("k"), py::arg("n"),
          py::arg("transpose_b") = false);
    m.def("softmax", &softmax, py::arg("x"));
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
