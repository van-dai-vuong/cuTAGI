#include "../include/tagi_autograd.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <unordered_set>

#include "../include/common.h"
#include "../include/linear_layer.h"
#include "../include/param_init.h"

namespace tagi_autograd {

namespace {
constexpr float EPS = 1e-12f;  // floor for POSTERIOR variances only
bool g_trace = false;
std::map<std::string, int> g_op_counters;

std::string autoname(const std::string& op) {
    int count = ++g_op_counters[op];
    return op + std::to_string(count);
}

std::string fmt(const std::vector<float>& v) {
    if (v.size() == 1) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%+.4f", v[0]);
        return buf;
    }
    float max_abs = 0.0f;
    for (float x : v) max_abs = std::max(max_abs, std::abs(x));
    char buf[64];
    std::snprintf(buf, sizeof(buf), "[n=%zu max|.|=%.4f]", v.size(), max_abs);
    return buf;
}

// Exact elementwise a/b where b > 0, and 0 where b == 0. Gain denominators
// must NOT be floored at some epsilon: a TAGI gain cov(parent,child)/var(child)
// is a Kalman gain -- the numerator always shrinks together with the
// denominator, so the exact ratio is bounded, while flooring the denominator
// silently attenuates updates through low-variance nodes (e.g. saturated
// sigmoids where var(a) = J^2 var(z) ~ 1e-14) by orders of magnitude.
std::vector<float> safe_div(const std::vector<float>& a,
                            const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out[i] = b[i] > 0.0f ? a[i] / b[i] : 0.0f;
    }
    return out;
}

void check_fresh(std::initializer_list<TensorPtr> nodes) {
    for (const auto& n : nodes) {
        if (n->used) {
            throw std::runtime_error(
                "node '" + n->name +
                "' belongs to an already-conditioned graph; its stored "
                "moments are stale priors. Re-run the forward pass.");
        }
    }
}

}  // namespace

void set_trace(bool enabled) { g_trace = enabled; }

// =============================================================================
//  GaussianTensor
// =============================================================================
GaussianTensor::GaussianTensor(std::vector<float> mu_in,
                               std::vector<float> var_in,
                               std::vector<int> shape_in, std::string name_in)
    : mu(std::move(mu_in)),
      var(std::move(var_in)),
      shape(std::move(shape_in)),
      name(std::move(name_in)) {}

GaussianTensor& GaussianTensor::named(const std::string& n) {
    name = n;
    return *this;
}

GaussianTensor& GaussianTensor::retain() {
    retain_flag = true;
    return *this;
}

TensorPtr GaussianTensor::detach() const {
    const std::vector<float>& src_mu = has_post ? post_mu : mu;
    const std::vector<float>& src_var = has_post ? post_var : var;
    return std::make_shared<GaussianTensor>(src_mu, src_var, shape,
                                            name + ".detached");
}

void GaussianTensor::accumulate(const std::vector<float>& d_mu_in,
                                const std::vector<float>& d_var_in) {
    if (g_trace) {
        std::printf("      OUT -> %s._accumulate(d_mu=%s, d_var=%s)\n",
                    name.c_str(), fmt(d_mu_in).c_str(), fmt(d_var_in).c_str());
    }
    if (!has_innovation) {
        d_mu.assign(mu.size(), 0.0f);
        d_var.assign(var.size(), 0.0f);
        has_innovation = true;
    }
    for (size_t i = 0; i < d_mu.size(); ++i) {
        d_mu[i] += d_mu_in[i];
        d_var[i] += d_var_in[i];
    }
}

void GaussianTensor::observe(const std::vector<float>& y, float var_v) {
    if (used) {
        throw std::runtime_error(
            "observe(): this graph was already conditioned; parameters "
            "were updated so its cached priors are stale. Re-run the "
            "forward pass to build a fresh graph.");
    }
    if (g_trace) {
        std::printf("[SEED]    observe(y=%s, var_v=%g) on '%s'\n",
                    fmt(y).c_str(), var_v, name.c_str());
    }
    std::vector<float> var_y(var.size());
    for (size_t i = 0; i < var.size(); ++i) var_y[i] = var[i] + var_v;
    std::vector<float> gain = safe_div(var, var_y);

    std::vector<float> d_mu_seed(mu.size()), d_var_seed(var.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        d_mu_seed[i] = gain[i] * (y[i] - mu[i]);
        d_var_seed[i] = -gain[i] * var[i];
    }
    accumulate(d_mu_seed, d_var_seed);
    backward();
}

void GaussianTensor::backward() {
    std::vector<GaussianTensor*> topo;
    std::unordered_set<GaussianTensor*> seen;

    std::function<void(GaussianTensor*)> dfs = [&](GaussianTensor* node) {
        if (seen.count(node)) return;
        seen.insert(node);
        for (auto& p : node->parents) dfs(p.get());
        topo.push_back(node);
    };
    dfs(this);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        GaussianTensor* node = *it;
        Parameter* param = dynamic_cast<Parameter*>(node);
        if (param == nullptr) node->used = true;  // Parameters ARE reusable

        if (node->retain_flag) {
            node->post_mu = node->mu;
            node->post_var = node->var;
            if (node->has_innovation) {
                for (size_t i = 0; i < node->post_mu.size(); ++i) {
                    node->post_mu[i] += node->d_mu[i];
                    node->post_var[i] += node->d_var[i];
                }
            }
            for (auto& v : node->post_var) v = std::max(v, EPS);
            node->has_post = true;
        }

        if (!node->has_innovation) continue;

        if (node->backward_fn) {
            if (g_trace) {
                std::printf("[CLOSURE] %s._backward  IN: d_mu=%s, d_var=%s\n",
                            node->name.c_str(), fmt(node->d_mu).c_str(),
                            fmt(node->d_var).c_str());
            }
            node->backward_fn(node->d_mu, node->d_var);
        }
        if (param != nullptr) param->apply_update();

        // Free innovations (graph is per-forward, like autograd).
        node->has_innovation = false;
        node->d_mu.clear();
        node->d_var.clear();
    }
}

// =============================================================================
//  Parameter
// =============================================================================
Parameter::Parameter(std::vector<float> mu, std::vector<float> var,
                     std::vector<int> shape, std::string name)
    : GaussianTensor(std::move(mu), std::move(var), std::move(shape),
                     std::move(name)) {}

void Parameter::apply_update() {
    if (g_trace) {
        std::vector<float> new_mu(mu.size()), new_var(var.size());
        for (size_t i = 0; i < mu.size(); ++i) new_mu[i] = mu[i] + d_mu[i];
        for (size_t i = 0; i < var.size(); ++i)
            new_var[i] = std::max(var[i] + d_var[i], EPS);
        std::printf("[SINK]    %s: mu %s -> %s, var %s -> %s\n", name.c_str(),
                    fmt(mu).c_str(), fmt(new_mu).c_str(), fmt(var).c_str(),
                    fmt(new_var).c_str());
    }
    for (size_t i = 0; i < mu.size(); ++i) mu[i] += d_mu[i];
    for (size_t i = 0; i < var.size(); ++i)
        var[i] = std::max(var[i] + d_var[i], EPS);
}

TensorPtr tensor(std::vector<float> mu, std::vector<int> shape,
                 std::vector<float> var, std::string name) {
    if (var.empty()) var.assign(mu.size(), 0.0f);
    return std::make_shared<GaussianTensor>(std::move(mu), std::move(var),
                                            std::move(shape), std::move(name));
}

// =============================================================================
//  Op 1: ADDITION of independent Gaussians
// =============================================================================
TensorPtr add(const TensorPtr& x, const TensorPtr& y) {
    check_fresh({x, y});
    if (x->mu.size() != y->mu.size()) {
        throw std::invalid_argument(
            "add: shape mismatch; broadcast explicitly so innovations are "
            "well-defined");
    }
    int n = static_cast<int>(x->mu.size());
    std::vector<float> mu(n), var(n);
    for (int i = 0; i < n; ++i) {
        mu[i] = x->mu[i] + y->mu[i];
        var[i] = x->var[i] + y->var[i];  // independence
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname("add"));
    out->parents = {x, y};
    out->backward_fn = [x, y, var](const std::vector<float>& d_mu,
                                   const std::vector<float>& d_var) {
        for (const auto& p : {x, y}) {
            std::vector<float> j = safe_div(p->var, var);  // var(p)/var(z)
            std::vector<float> pd_mu(j.size()), pd_var(j.size());
            for (size_t i = 0; i < j.size(); ++i) {
                pd_mu[i] = j[i] * d_mu[i];
                pd_var[i] = j[i] * j[i] * d_var[i];
            }
            p->accumulate(pd_mu, pd_var);
        }
    };
    return out;
}

// =============================================================================
//  Op 2: element-wise MULTIPLICATION (GMA, independent factors)
// =============================================================================
TensorPtr mul(const TensorPtr& x, const TensorPtr& y) {
    check_fresh({x, y});
    if (x->mu.size() != y->mu.size()) {
        throw std::invalid_argument(
            "mul: shape mismatch; broadcast explicitly so innovations are "
            "well-defined");
    }
    if (x.get() == y.get()) {
        throw std::invalid_argument(
            "mul(x, x): GMA assumes independent factors; x*x needs the "
            "correlated-GMA terms (cov(X1,X2)!=0)");
    }
    int n = static_cast<int>(x->mu.size());
    std::vector<float> mu(n), var(n);
    for (int i = 0; i < n; ++i) {
        mu[i] = x->mu[i] * y->mu[i];
        var[i] = x->var[i] * y->var[i] + x->var[i] * y->mu[i] * y->mu[i] +
                 y->var[i] * x->mu[i] * x->mu[i];
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname("mul"));
    out->parents = {x, y};
    out->backward_fn = [x, y, var](const std::vector<float>& d_mu,
                                   const std::vector<float>& d_var) {
        int n = static_cast<int>(d_mu.size());
        std::vector<float> jx(n), jy(n);
        for (int i = 0; i < n; ++i) {
            jx[i] = var[i] > 0.0f ? (x->var[i] * y->mu[i]) / var[i] : 0.0f;
            jy[i] = var[i] > 0.0f ? (y->var[i] * x->mu[i]) / var[i] : 0.0f;
        }
        std::vector<float> xd_mu(n), xd_var(n), yd_mu(n), yd_var(n);
        for (int i = 0; i < n; ++i) {
            xd_mu[i] = jx[i] * d_mu[i];
            xd_var[i] = jx[i] * jx[i] * d_var[i];
            yd_mu[i] = jy[i] * d_mu[i];
            yd_var[i] = jy[i] * jy[i] * d_var[i];
        }
        x->accumulate(xd_mu, xd_var);
        y->accumulate(yd_mu, yd_var);
    };
    return out;
}

// =============================================================================
//  Op 3: ACTIVATIONS -- each computes (mu_a, var_a, jcb) per element from
//  (index, mu_z, var_z), matching its namesake *_mean_var function in
//  src/activation.cpp exactly. cov(z,a) = jcb * var(z) always holds (true
//  by construction for the linearized ones; true by the TAGI derivation
//  for the closed-form truncated-/rectified-Gaussian ones too), so one
//  backward closure covers every activation below.
// =============================================================================
namespace {
using ElementwiseActivationFn = std::function<void(
    int i, float mu_z, float var_z, float& mu_a, float& var_a, float& jcb)>;

TensorPtr activation_op(const TensorPtr& x, const ElementwiseActivationFn& fn,
                        const std::string& op_name) {
    check_fresh({x});
    int n = static_cast<int>(x->mu.size());
    std::vector<float> mu(n), var(n), cov(n);
    for (int i = 0; i < n; ++i) {
        float jcb;
        fn(i, x->mu[i], x->var[i], mu[i], var[i], jcb);
        cov[i] = jcb * x->var[i];  // cov(Z, A) = jcb * var(Z)
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname(op_name));
    out->parents = {x};
    out->backward_fn = [x, cov, var](const std::vector<float>& d_mu,
                                     const std::vector<float>& d_var) {
        std::vector<float> j = safe_div(cov, var);  // exact 1/jcb where var>0
        std::vector<float> pd_mu(j.size()), pd_var(j.size());
        for (size_t i = 0; i < j.size(); ++i) {
            pd_mu[i] = j[i] * d_mu[i];
            pd_var[i] = j[i] * j[i] * d_var[i];
        }
        x->accumulate(pd_mu, pd_var);
    };
    return out;
}
}  // namespace

TensorPtr relu(const TensorPtr& x) {
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            float tmp = std::max(mu_z, 0.0f);
            mu_a = tmp;
            if (tmp == 0.0f) {
                jcb = 0.0f;
                var_a = 0.0f;
            } else {
                jcb = 1.0f;
                var_a = var_z;
            }
        },
        "relu");
}

TensorPtr tanh_act(const TensorPtr& x) {
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            float tmp = std::tanh(mu_z);
            mu_a = tmp;
            jcb = 1.0f - tmp * tmp;
            var_a = jcb * var_z * jcb;
        },
        "tanh");
}

TensorPtr sigmoid(const TensorPtr& x) {
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            float tmp = 1.0f / (1.0f + std::exp(-mu_z));
            mu_a = tmp;
            jcb = tmp * (1.0f - tmp);
            var_a = jcb * var_z * jcb;
        },
        "sigmoid");
}

TensorPtr mixture_relu(const TensorPtr& x) {
    constexpr float kSqrt2Pi = 2.5066282746310002f;
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            float std_z = std::sqrt(var_z);
            float alpha = mu_z / std_z;
            float pdf_alpha =
                (1.0f / kSqrt2Pi) * std::exp(-0.5f * alpha * alpha);
            float cdf_alpha = normcdf_cpu(alpha);

            // Moments calculations (L. Alric, 2024)
            float tmp_mu_a = mu_z * cdf_alpha + std_z * pdf_alpha;
            mu_a = std::max(0.000001f, tmp_mu_a);
            var_a =
                std::max(0.000001f, -tmp_mu_a * tmp_mu_a + 2 * tmp_mu_a * mu_z -
                                        mu_z * std_z * pdf_alpha +
                                        (var_z - mu_z * mu_z) * cdf_alpha);
            jcb = cdf_alpha;
        },
        "mixture_relu");
}

TensorPtr mixture_sigmoid(const TensorPtr& x) {
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            float std_z = std::sqrt(var_z);
            float alpha_l = (1.0f + mu_z) / std_z;  // lower truncation
            float alpha_u = (1.0f - mu_z) / std_z;  // upper truncation
            float cdf_l = normcdf_cpu(alpha_l);
            float cdf_u = normcdf_cpu(alpha_u);
            float pdf_l = normpdf_cpu(alpha_l, 0.0f, 1.0f);
            float pdf_u = normpdf_cpu(alpha_u, 0.0f, 1.0f);

            // Moments calculations (L. Alric, 2024)
            mu_a = (mu_z + 1) * cdf_l + (mu_z - 1) * cdf_u +
                   std_z * (pdf_l - pdf_u) - mu_z;
            var_a = std::max(
                0.000001f,
                (cdf_l * (var_z - mu_z * mu_z - 2 * mu_z - 1) +
                 cdf_u * (var_z - mu_z * mu_z + 2 * mu_z - 1) +
                 std_z * (pdf_u * (mu_z - 1) - pdf_l * (mu_z + 1)) -
                 mu_a * mu_a + 2 * mu_a * mu_z + mu_z * mu_z - var_z + 2) /
                    4.0f);
            mu_a = mu_a / 2.0f + 0.5f;
            jcb = (cdf_u + cdf_l - 1.0f) / 2.0f;
        },
        "mixture_sigmoid");
}

TensorPtr mixture_tanh(const TensorPtr& x) {
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            float std_z = std::sqrt(var_z);
            float alpha_l = (1.0f + mu_z) / std_z;  // lower truncation
            float alpha_u = (1.0f - mu_z) / std_z;  // upper truncation
            float cdf_l = normcdf_cpu(alpha_l);
            float cdf_u = normcdf_cpu(alpha_u);
            float pdf_l = normpdf_cpu(alpha_l, 0.0f, 1.0f);
            float pdf_u = normpdf_cpu(alpha_u, 0.0f, 1.0f);

            // Moments calculations (L. Alric, 2024)
            mu_a = (mu_z + 1) * cdf_l + (mu_z - 1) * cdf_u +
                   std_z * (pdf_l - pdf_u) - mu_z;
            var_a = std::max(
                0.000001f,
                cdf_l * (var_z - mu_z * mu_z - 2 * mu_z - 1) +
                    cdf_u * (var_z - mu_z * mu_z + 2 * mu_z - 1) +
                    std_z * (pdf_u * (mu_z - 1) - pdf_l * (mu_z + 1)) -
                    mu_a * mu_a + 2 * mu_a * mu_z + mu_z * mu_z - var_z + 2);
            jcb = cdf_u + cdf_l - 1.0f;
        },
        "mixture_tanh");
}

TensorPtr softplus(const TensorPtr& x) {
    return activation_op(
        x,
        [](int, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            mu_a = std::log(1.0f + std::exp(mu_z));
            float tmp = 1.0f / (1.0f + std::exp(-mu_z));
            jcb = tmp;
            var_a = tmp * var_z * tmp;
        },
        "softplus");
}

TensorPtr leaky_relu(const TensorPtr& x, float alpha) {
    return activation_op(
        x,
        [alpha](int, float mu_z, float var_z, float& mu_a, float& var_a,
                float& jcb) {
            float tmp = std::max(mu_z, 0.0f);
            if (tmp == 0.0f) {
                mu_a = alpha * mu_z;
                jcb = alpha;
                var_a = alpha * var_z * alpha;
            } else {
                mu_a = tmp;
                jcb = 1.0f;
                var_a = var_z;
            }
        },
        "leaky_relu");
}

TensorPtr even_exp(const TensorPtr& x) {
    return activation_op(
        x,
        [](int i, float mu_z, float var_z, float& mu_a, float& var_a,
           float& jcb) {
            if (i % 2 == 0) {
                mu_a = mu_z;
                var_a = var_z;
                jcb = 1.0f;  // pass-through: cov(z,a) = var(z)
            } else {
                mu_a = std::exp(mu_z + 0.5f * var_z);
                var_a = std::exp(2 * mu_z + var_z) * (std::exp(var_z) - 1.0f);
                jcb = mu_a;  // cov(Z, exp(Z)) = var(Z) * mu_exp(Z)
            }
        },
        "even_exp");
}

// =============================================================================
//  chunk: split a tensor into equal parts along the last axis. Each chunk
//  IS a slice of its parent, so the backward gain is exactly 1.
// =============================================================================
std::vector<TensorPtr> chunk(const TensorPtr& x, int chunks,
                             const std::string& name_in) {
    check_fresh({x});
    std::string name = name_in.empty() ? autoname("chunk") : name_in;
    int rows = x->shape[0];
    int cols = x->shape[1];
    if (cols % chunks != 0) {
        throw std::invalid_argument("chunk: last dim " + std::to_string(cols) +
                                    " not divisible by " +
                                    std::to_string(chunks));
    }
    int size = cols / chunks;

    std::vector<TensorPtr> outs;
    outs.reserve(chunks);
    for (int k = 0; k < chunks; ++k) {
        std::vector<float> mu(rows * size), var(rows * size);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < size; ++c) {
                mu[r * size + c] = x->mu[r * cols + k * size + c];
                var[r * size + c] = x->var[r * cols + k * size + c];
            }
        }
        auto out = std::make_shared<GaussianTensor>(
            mu, var, std::vector<int>{rows, size},
            name + "[" + std::to_string(k) + "]");
        out->parents = {x};
        out->backward_fn = [x, rows, cols, size, k](
                               const std::vector<float>& d_mu,
                               const std::vector<float>& d_var) {
            std::vector<float> full_mu(rows * cols, 0.0f);
            std::vector<float> full_var(rows * cols, 0.0f);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < size; ++c) {
                    full_mu[r * cols + k * size + c] = d_mu[r * size + c];
                    full_var[r * cols + k * size + c] = d_var[r * size + c];
                }
            }
            x->accumulate(full_mu, full_var);
        };
        outs.push_back(out);
    }
    return outs;
}

// =============================================================================
//  Affine map z = x W^T + b
// =============================================================================
TensorPtr linear(const TensorPtr& x, const ParamPtr& w, const ParamPtr& b) {
    check_fresh({x});
    int batch = x->shape[0];
    int n_in = x->shape[1];
    int n_out = w->shape[0];

    // Forward moments come from the production kernel in
    // src/linear_layer.cpp (the one the hand-written-backward Linear layer
    // uses) -- the memory layouts already agree: x is (batch, n_in)
    // row-major, w is (n_out, n_in) row-major, z is (batch, n_out)
    // row-major. [0, n_out * batch) covers the whole output in one chunk.
    std::vector<float> mu(batch * n_out, 0.0f), var(batch * n_out, 0.0f);
    std::vector<float> no_bias;  // untouched when bias == false
    ::linear_fwd_mean_var(w->mu, w->var, b != nullptr ? b->mu : no_bias,
                          b != nullptr ? b->var : no_bias, x->mu, x->var,
                          /*start_chunk=*/0, /*end_chunk=*/n_out * batch, n_in,
                          n_out, batch, /*bias=*/b != nullptr, mu, var);

    auto out = std::make_shared<GaussianTensor>(
        mu, var, std::vector<int>{batch, n_out}, autoname("linear"));
    out->parents = (b != nullptr) ? std::vector<TensorPtr>{x, w, b}
                                  : std::vector<TensorPtr>{x, w};

    out->backward_fn = [x, w, b, batch, n_in, n_out, var](
                           const std::vector<float>& d_mu,
                           const std::vector<float>& d_var) {
        const std::vector<float>& mu_x = x->mu;
        const std::vector<float>& var_x = x->var;
        const std::vector<float>& mu_w = w->mu;
        const std::vector<float>& var_w = w->var;

        std::vector<float> r_mu(batch * n_out), r_var(batch * n_out);
        for (int i = 0; i < batch * n_out; ++i) {
            r_mu[i] = var[i] > 0.0f ? d_mu[i] / var[i] : 0.0f;
            r_var[i] = var[i] > 0.0f ? d_var[i] / (var[i] * var[i]) : 0.0f;
        }

        // ---- innovations for the input hidden states ----
        //   d_mu_x_k  = sum_i cov(x_k,z_i)/var(z_i) d_mu_i
        std::vector<float> xd_mu(batch * n_in, 0.0f);
        std::vector<float> xd_var(batch * n_in, 0.0f);
        for (int r = 0; r < batch; ++r) {
            for (int k = 0; k < n_in; ++k) {
                float sum_mu = 0.0f, sum_var = 0.0f;
                for (int o = 0; o < n_out; ++o) {
                    float mwok = mu_w[o * n_in + k];
                    sum_mu += r_mu[r * n_out + o] * mwok;
                    sum_var += r_var[r * n_out + o] * mwok * mwok;
                }
                float vxk = var_x[r * n_in + k];
                xd_mu[r * n_in + k] = sum_mu * vxk;
                xd_var[r * n_in + k] = sum_var * vxk * vxk;
            }
        }
        x->accumulate(xd_mu, xd_var);

        // ---- innovations for the parameters (SUMMED over the batch, like
        // production linear_bwd_fc_delta_w: each sample is an independent
        // observation of the same weights, so their updates accumulate;
        // averaging would shrink learning by a factor of batch_size) ----
        std::vector<float> dw_mu(n_out * n_in, 0.0f);
        std::vector<float> dw_var(n_out * n_in, 0.0f);
        for (int o = 0; o < n_out; ++o) {
            for (int k = 0; k < n_in; ++k) {
                float sum_mu = 0.0f, sum_var = 0.0f;
                for (int r = 0; r < batch; ++r) {
                    float mxk = mu_x[r * n_in + k];
                    sum_mu += r_mu[r * n_out + o] * mxk;
                    sum_var += r_var[r * n_out + o] * mxk * mxk;
                }
                dw_mu[o * n_in + k] = sum_mu;
                dw_var[o * n_in + k] = sum_var;
            }
        }
        std::vector<float> wd_mu(n_out * n_in), wd_var(n_out * n_in);
        for (int i = 0; i < n_out * n_in; ++i) {
            wd_mu[i] = var_w[i] * dw_mu[i];
            wd_var[i] = var_w[i] * var_w[i] * dw_var[i];
        }
        w->accumulate(wd_mu, wd_var);

        if (b != nullptr) {
            std::vector<float> db_mu(n_out, 0.0f), db_var(n_out, 0.0f);
            for (int o = 0; o < n_out; ++o) {
                for (int r = 0; r < batch; ++r) {
                    db_mu[o] += r_mu[r * n_out + o];
                    db_var[o] += r_var[r * n_out + o];
                }
            }
            std::vector<float> bd_mu(n_out), bd_var(n_out);
            for (int o = 0; o < n_out; ++o) {
                bd_mu[o] = b->var[o] * db_mu[o];
                bd_var[o] = b->var[o] * b->var[o] * db_var[o];
            }
            b->accumulate(bd_mu, bd_var);
        }
    };
    return out;
}

// =============================================================================
//  Module / Linear / Sequential
// =============================================================================
void Module::register_parameter(const ParamPtr& p) { params_.push_back(p); }

void Module::register_module(const std::shared_ptr<Module>& m) {
    submodules_.push_back(m);
}

std::vector<ParamPtr> Module::parameters() const {
    std::vector<ParamPtr> out = params_;
    for (const auto& m : submodules_) {
        auto sub = m->parameters();
        out.insert(out.end(), sub.begin(), sub.end());
    }
    return out;
}

Linear::Linear(int n_in, int n_out, bool bias_, float gain_w, float gain_b,
               const std::string& init_method, int fan_in_override) {
    int fan_in = fan_in_override > 0 ? fan_in_override : n_in;
    int num_weights = n_out * n_in;
    int num_biases = bias_ ? n_out : 0;

    std::vector<float> mu_w, var_w, mu_b, var_b;
    std::tie(mu_w, var_w, mu_b, var_b) = init_weight_bias_linear(
        init_method, gain_w, gain_b, fan_in, n_out, num_weights, num_biases);

    weight = std::make_shared<Parameter>(mu_w, var_w,
                                         std::vector<int>{n_out, n_in}, "W");
    register_parameter(weight);

    if (bias_) {
        bias = std::make_shared<Parameter>(mu_b, var_b,
                                           std::vector<int>{1, n_out}, "b");
        register_parameter(bias);
    }
}

TensorPtr Linear::forward(const TensorPtr& x) {
    return linear(x, weight, bias);
}

FunctionModule::FunctionModule(std::function<TensorPtr(const TensorPtr&)> fn)
    : fn_(std::move(fn)) {}

TensorPtr FunctionModule::forward(const TensorPtr& x) { return fn_(x); }

std::shared_ptr<Module> relu_module() {
    return std::make_shared<FunctionModule>(
        [](const TensorPtr& x) { return relu(x); });
}
std::shared_ptr<Module> tanh_module() {
    return std::make_shared<FunctionModule>(
        [](const TensorPtr& x) { return tanh_act(x); });
}
std::shared_ptr<Module> sigmoid_module() {
    return std::make_shared<FunctionModule>(
        [](const TensorPtr& x) { return sigmoid(x); });
}

Sequential::Sequential(std::vector<std::shared_ptr<Module>> items_)
    : items(std::move(items_)) {
    for (const auto& m : items) register_module(m);
}

TensorPtr Sequential::forward(const TensorPtr& x) {
    TensorPtr out = x;
    for (const auto& m : items) out = m->forward(out);
    return out;
}

}  // namespace tagi_autograd
