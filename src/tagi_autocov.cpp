#include "../include/tagi_autocov.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <unordered_set>

#include "../include/activation.h"
#include "../include/common.h"
#include "../include/linear_layer.h"
#include "../include/param_init.h"

namespace tagi_autocov {

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

// ---------------------------------------------------------------------------
// Covariance tape (see the block comment in tagi_autocov.h). Weak registry
// of every node that currently owns a tape, so clear_cov_tapes() can free
// them without keeping any graph alive.
// ---------------------------------------------------------------------------
std::vector<std::weak_ptr<GaussianTensor>> g_taped_nodes;

void register_taped(const TensorPtr& t) {
    if (!t->cov_tape.empty()) g_taped_nodes.push_back(t);
}

const GaussianTensor::CovTapeEntry* find_tape_entry(
    const GaussianTensor& node, const GaussianTensor* root) {
    for (const auto& e : node.cov_tape) {
        if (e.root == root) return &e;
    }
    return nullptr;
}

// DIAGONAL cross-covariance rho_i = cov(x_i, y_i) reconstructed from the
// roots BOTH tapes share:
//     rho = sum_r [J_x diag(var_r) J_y^T]_ii
// This is the fast path add()/mul() consume: they pair same-shape operands
// element by element, so only the diagonal of the full matrix is needed;
// the full (numel_x, numel_y) matrix is the public cross_cov() below.
// Roots are ancestors of x/y, so their pointers are kept alive by the
// parents chain for the duration of this (forward-time) call. Returns an
// empty vector when x and y share no roots -- callers treat that as the
// independent case, at zero cost. Each rho_i is clamped to the
// Cauchy-Schwarz bound +/- sqrt(var_x var_y): the bound holds exactly for
// linearized activations, but the closed-form mixture_* moments can
// slightly exceed it.
std::vector<float> cross_cov_diag(const GaussianTensor& x,
                                  const GaussianTensor& y) {
    std::vector<float> rho;
    int n = static_cast<int>(x.mu.size());
    int cols = x.shape[1];
    for (const auto& ex : x.cov_tape) {
        const auto* ey = find_tape_entry(y, ex.root);
        if (ey == nullptr) continue;
        if (rho.empty()) rho.assign(n, 0.0f);
        int rc = ex.root_cols;
        const std::vector<float>& var_r = ex.root->var;
        for (int i = 0; i < n; ++i) {
            const float* jx = &ex.jac[static_cast<size_t>(i) * rc];
            const float* jy = &ey->jac[static_cast<size_t>(i) * rc];
            const float* vr = &var_r[static_cast<size_t>(i / cols) * rc];
            float s = 0.0f;
            for (int k = 0; k < rc; ++k) s += jx[k] * vr[k] * jy[k];
            rho[i] += s;
        }
    }
    if (!rho.empty()) {
        for (int i = 0; i < n; ++i) {
            float bound = std::sqrt(x.var[i] * y.var[i]);
            rho[i] = std::max(-bound, std::min(bound, rho[i]));
        }
    }
    return rho;
}

// J_out = diag(g) J_x -- elementwise ops (activations, scale).
void tape_unary(const TensorPtr& x, const std::vector<float>& g,
                const TensorPtr& out) {
    if (x->cov_tape.empty()) return;
    for (const auto& ex : x->cov_tape) {
        GaussianTensor::CovTapeEntry e{ex.root, ex.root_cols,
                                       std::vector<float>(ex.jac.size())};
        int rc = ex.root_cols;
        for (size_t i = 0; i < g.size(); ++i) {
            for (int k = 0; k < rc; ++k) {
                e.jac[i * rc + k] = g[i] * ex.jac[i * rc + k];
            }
        }
        out->cov_tape.push_back(std::move(e));
    }
    register_taped(out);
}

// J_out = diag(gx) J_x + diag(gy) J_y over the union of both root sets;
// gx/gy == nullptr means all-ones (add). For mul, gx = mu_y and gy = mu_x.
void tape_combine(const TensorPtr& x, const std::vector<float>* gx,
                  const TensorPtr& y, const std::vector<float>* gy,
                  const TensorPtr& out) {
    if (x->cov_tape.empty() && y->cov_tape.empty()) return;
    size_t n = x->mu.size();
    auto accumulate_scaled = [n](GaussianTensor::CovTapeEntry& e,
                                 const GaussianTensor::CovTapeEntry& src,
                                 const std::vector<float>* g) {
        int rc = src.root_cols;
        for (size_t i = 0; i < n; ++i) {
            float gi = g != nullptr ? (*g)[i] : 1.0f;
            for (int k = 0; k < rc; ++k) {
                e.jac[i * rc + k] += gi * src.jac[i * rc + k];
            }
        }
    };
    for (const auto& ex : x->cov_tape) {
        GaussianTensor::CovTapeEntry e{ex.root, ex.root_cols,
                                       std::vector<float>(ex.jac.size(), 0.0f)};
        accumulate_scaled(e, ex, gx);
        if (const auto* ey = find_tape_entry(*y, ex.root)) {
            accumulate_scaled(e, *ey, gy);
        }
        out->cov_tape.push_back(std::move(e));
    }
    for (const auto& ey : y->cov_tape) {
        if (find_tape_entry(*x, ey.root) != nullptr) continue;  // merged above
        GaussianTensor::CovTapeEntry e{ey.root, ey.root_cols,
                                       std::vector<float>(ey.jac.size(), 0.0f)};
        accumulate_scaled(e, ey, gy);
        out->cov_tape.push_back(std::move(e));
    }
    register_taped(out);
}

// J_z = mu_W J_x per batch row -- z = x W^T (+ b). The weights and bias
// are node-private parameters: they add variance to z but no sensitivity
// to any shared root, so they never appear on the tape.
void tape_linear(const TensorPtr& x, const std::vector<float>& mu_w, int batch,
                 int n_in, int n_out, const TensorPtr& out) {
    if (x->cov_tape.empty()) return;
    for (const auto& ex : x->cov_tape) {
        int rc = ex.root_cols;
        GaussianTensor::CovTapeEntry e{
            ex.root, rc,
            std::vector<float>(static_cast<size_t>(batch) * n_out * rc, 0.0f)};
        for (int b = 0; b < batch; ++b) {
            for (int o = 0; o < n_out; ++o) {
                float* jz = &e.jac[(static_cast<size_t>(b) * n_out + o) * rc];
                for (int j = 0; j < n_in; ++j) {
                    float w = mu_w[o * n_in + j];
                    if (w == 0.0f) continue;
                    const float* jx =
                        &ex.jac[(static_cast<size_t>(b) * n_in + j) * rc];
                    for (int k = 0; k < rc; ++k) jz[k] += w * jx[k];
                }
            }
        }
        out->cov_tape.push_back(std::move(e));
    }
    register_taped(out);
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

GaussianTensor& GaussianTensor::track_cov() {
    int rows = shape[0];
    int cols = shape[1];
    // Reset to {self: I}: this node becomes an exogenous Gaussian source
    // with its current marginal variance (see header). Discarding any
    // inherited entries prevents double counting -- the variance those
    // roots explain is already inside this node's var.
    CovTapeEntry e{
        this, cols,
        std::vector<float>(static_cast<size_t>(rows) * cols * cols, 0.0f)};
    for (int b = 0; b < rows; ++b) {
        for (int c = 0; c < cols; ++c) {
            e.jac[(static_cast<size_t>(b) * cols + c) * cols + c] = 1.0f;
        }
    }
    cov_tape.clear();
    cov_tape.push_back(std::move(e));
    g_taped_nodes.push_back(shared_from_this());
    return *this;
}

void clear_cov_tapes() {
    for (auto& w : g_taped_nodes) {
        if (auto p = w.lock()) {
            std::vector<GaussianTensor::CovTapeEntry>().swap(p->cov_tape);
        }
    }
    g_taped_nodes.clear();
}

std::vector<float> cross_cov(const TensorPtr& x, const TensorPtr& y) {
    // Per-batch-row covariance blocks: block b is the (cols_x, cols_y)
    // matrix cov(x[b,i], y[b,j]) = sum_r [J_x diag(var_r) J_y^T]_ij, flat
    // (rows * cols_x * cols_y). Batch rows are independent throughout the
    // engine, so covariance between different batch rows is structurally
    // zero and is NOT stored -- for {16,8} nodes the result is 16 blocks
    // of (8,8), not a mostly-zero (128,128) matrix. x and y may have
    // different column counts (e.g. a root x {1,3} against z = linear(x)
    // {1,5} gives the (3, 5) block diag(var_x) mu_W^T). Nodes sharing a
    // root always have the root's row count, so blocks align.
    int rows = x->shape[0];
    int cx = x->shape[1], cy = y->shape[1];
    std::vector<float> cov;
    for (const auto& ex : x->cov_tape) {
        const auto* ey = find_tape_entry(*y, ex.root);
        if (ey == nullptr) continue;
        if (cov.empty()) {
            cov.assign(static_cast<size_t>(rows) * cx * cy, 0.0f);
        }
        int rc = ex.root_cols;
        const std::vector<float>& var_r = ex.root->var;
        for (int b = 0; b < rows; ++b) {
            const float* vr = &var_r[static_cast<size_t>(b) * rc];
            for (int i = 0; i < cx; ++i) {
                const float* jx =
                    &ex.jac[(static_cast<size_t>(b) * cx + i) * rc];
                for (int j = 0; j < cy; ++j) {
                    const float* jy =
                        &ey->jac[(static_cast<size_t>(b) * cy + j) * rc];
                    float s = 0.0f;
                    for (int k = 0; k < rc; ++k) s += jx[k] * vr[k] * jy[k];
                    cov[(static_cast<size_t>(b) * cx + i) * cy + j] += s;
                }
            }
        }
    }
    if (!cov.empty()) {
        // Same Cauchy-Schwarz clamp as the diagonal path, per pair.
        for (int b = 0; b < rows; ++b) {
            for (int i = 0; i < cx; ++i) {
                for (int j = 0; j < cy; ++j) {
                    float bound =
                        std::sqrt(x->var[b * cx + i] * y->var[b * cy + j]);
                    float& c = cov[(static_cast<size_t>(b) * cx + i) * cy + j];
                    c = std::max(-bound, std::min(bound, c));
                }
            }
        }
    }
    return cov;
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
    // Normalized innovation-vector seed (production DeltaStates
    // convention): d_mu = (y - mu)/(var + var_v), d_var = -1/(var + var_v)
    // -- the raw innovations divided by this node's own variance. This is
    // the ONLY division in the whole backward pass: every op then chains
    // pure multiplications, and the consumers (Parameter::apply_update,
    // retain()) multiply by their own variance once at the end.
    std::vector<float> d_mu_seed(mu.size()), d_var_seed(var.size());
    for (size_t i = 0; i < mu.size(); ++i) {
        float var_y = var[i] + var_v;
        d_mu_seed[i] = var_y > 0.0f ? (y[i] - mu[i]) / var_y : 0.0f;
        d_var_seed[i] = var_y > 0.0f ? -1.0f / var_y : 0.0f;
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
                // d_mu/d_var are normalized by this node's own variance;
                // the raw posterior shift is var*d_mu / var^2*d_var.
                for (size_t i = 0; i < node->post_mu.size(); ++i) {
                    float v = node->var[i];
                    node->post_mu[i] += v * node->d_mu[i];
                    node->post_var[i] += v * v * node->d_var[i];
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
    // d_mu/d_var are normalized innovations: the parameter multiplies by
    // its OWN variance exactly once here (mu += var*d_mu,
    // var += var^2*d_var) -- the production delta_w convention.
    if (g_trace) {
        std::vector<float> new_mu(mu.size()), new_var(var.size());
        for (size_t i = 0; i < mu.size(); ++i)
            new_mu[i] = mu[i] + var[i] * d_mu[i];
        for (size_t i = 0; i < var.size(); ++i)
            new_var[i] = std::max(var[i] + var[i] * var[i] * d_var[i], EPS);
        std::printf("[SINK]    %s: mu %s -> %s, var %s -> %s\n", name.c_str(),
                    fmt(mu).c_str(), fmt(new_mu).c_str(), fmt(var).c_str(),
                    fmt(new_var).c_str());
    }
    for (size_t i = 0; i < mu.size(); ++i) mu[i] += var[i] * d_mu[i];
    for (size_t i = 0; i < var.size(); ++i)
        var[i] = std::max(var[i] + var[i] * var[i] * d_var[i], EPS);
}

TensorPtr tensor(std::vector<float> mu, std::vector<int> shape,
                 std::vector<float> var, std::string name) {
    if (var.empty()) var.assign(mu.size(), 0.0f);
    return std::make_shared<GaussianTensor>(std::move(mu), std::move(var),
                                            std::move(shape), std::move(name));
}

// =============================================================================
//  Op 1: ADDITION -- independent unless the covariance tape says otherwise
// =============================================================================
TensorPtr add(const TensorPtr& x, const TensorPtr& y) {
    check_fresh({x, y});
    if (x->mu.size() != y->mu.size()) {
        throw std::invalid_argument(
            "add: shape mismatch; broadcast explicitly so innovations are "
            "well-defined");
    }
    int n = static_cast<int>(x->mu.size());
    // rho = cov(x, y) from shared tape roots; empty means independent.
    std::vector<float> rho = cross_cov_diag(*x, *y);
    std::vector<float> mu(n), var(n);
    for (int i = 0; i < n; ++i) {
        mu[i] = x->mu[i] + y->mu[i];
        var[i] = x->var[i] + y->var[i];
        if (!rho.empty()) var[i] = std::max(var[i] + 2.0f * rho[i], 0.0f);
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname("add"));
    out->parents = {x, y};
    tape_combine(x, nullptr, y, nullptr, out);
    // Local factor g = cov(p,z)/var(p) = 1 + rho/var(p); with independent
    // parents it is exactly 1 and the normalized deltas pass through
    // unchanged. The rho/var ratio is computed here at FORWARD time (the
    // CS clamp guarantees rho = 0 wherever var = 0), so the backward pass
    // stays division-free.
    std::vector<float> gx, gy;
    if (!rho.empty()) {
        gx.resize(n);
        gy.resize(n);
        for (int i = 0; i < n; ++i) {
            gx[i] = 1.0f + (x->var[i] > 0.0f ? rho[i] / x->var[i] : 0.0f);
            gy[i] = 1.0f + (y->var[i] > 0.0f ? rho[i] / y->var[i] : 0.0f);
        }
    }
    out->backward_fn = [x, y, gx, gy](const std::vector<float>& d_mu,
                                      const std::vector<float>& d_var) {
        if (gx.empty()) {
            x->accumulate(d_mu, d_var);
            y->accumulate(d_mu, d_var);
            return;
        }
        auto push = [&](const TensorPtr& p, const std::vector<float>& g) {
            std::vector<float> pd_mu(d_mu.size()), pd_var(d_var.size());
            for (size_t i = 0; i < d_mu.size(); ++i) {
                pd_mu[i] = g[i] * d_mu[i];
                pd_var[i] = g[i] * g[i] * d_var[i];
            }
            p->accumulate(pd_mu, pd_var);
        };
        push(x, gx);
        push(y, gy);
    };
    return out;
}

// =============================================================================
//  Op 2: element-wise MULTIPLICATION (GMA) -- independent factors unless
//  the covariance tape supplies cov(x, y), in which case the exact moments
//  of a correlated Gaussian product are used (see header).
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
    // rho = cov(x, y) from shared tape roots; empty means independent.
    std::vector<float> rho = cross_cov_diag(*x, *y);
    std::vector<float> mu(n), var(n);
    for (int i = 0; i < n; ++i) {
        float r = rho.empty() ? 0.0f : rho[i];
        mu[i] = x->mu[i] * y->mu[i] + r;
        var[i] = x->var[i] * y->var[i] + x->var[i] * y->mu[i] * y->mu[i] +
                 y->var[i] * x->mu[i] * x->mu[i] + r * r +
                 2.0f * r * x->mu[i] * y->mu[i];
        var[i] = std::max(var[i], 0.0f);
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname("mul"));
    out->parents = {x, y};
    // d(xy)/dx = mu_y, d(xy)/dy = mu_x (prior means).
    tape_combine(x, &y->mu, y, &x->mu, out);
    // Local factors g = cov(p,z)/var(p): gx = mu_y + rho*mu_x/var_x and
    // symmetrically for y (plain mu_y/mu_x when independent). The
    // rho/var ratio is a FORWARD-time constant (the CS clamp guarantees
    // rho = 0 wherever var = 0), so the backward pass is division-free.
    std::vector<float> gx(n), gy(n);
    for (int i = 0; i < n; ++i) {
        float r = rho.empty() ? 0.0f : rho[i];
        gx[i] = y->mu[i] + (x->var[i] > 0.0f ? r * x->mu[i] / x->var[i] : 0.0f);
        gy[i] = x->mu[i] + (y->var[i] > 0.0f ? r * y->mu[i] / y->var[i] : 0.0f);
    }
    out->backward_fn = [x, y, gx, gy](const std::vector<float>& d_mu,
                                      const std::vector<float>& d_var) {
        int n = static_cast<int>(d_mu.size());
        std::vector<float> xd_mu(n), xd_var(n), yd_mu(n), yd_var(n);
        for (int i = 0; i < n; ++i) {
            xd_mu[i] = gx[i] * d_mu[i];
            xd_var[i] = gx[i] * gx[i] * d_var[i];
            yd_mu[i] = gy[i] * d_mu[i];
            yd_var[i] = gy[i] * gy[i] * d_var[i];
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
    std::vector<float> mu(n), var(n), jcb_v(n);
    for (int i = 0; i < n; ++i) {
        fn(i, x->mu[i], x->var[i], mu[i], var[i], jcb_v[i]);
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname(op_name));
    out->parents = {x};
    tape_unary(x, jcb_v, out);  // J_a = diag(jcb) J_z
    // Local factor g = cov(z,a)/var(z) = jcb -- the same multiplier the
    // production activation backward applies to its DeltaStates.
    out->backward_fn = [x, jcb_v](const std::vector<float>& d_mu,
                                  const std::vector<float>& d_var) {
        std::vector<float> pd_mu(d_mu.size()), pd_var(d_var.size());
        for (size_t i = 0; i < d_mu.size(); ++i) {
            pd_mu[i] = jcb_v[i] * d_mu[i];
            pd_var[i] = jcb_v[i] * jcb_v[i] * d_var[i];
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
        // Tape: a chunk element IS its source element, so its row of the
        // sensitivity matrix is copied verbatim.
        for (const auto& ex : x->cov_tape) {
            int rc = ex.root_cols;
            GaussianTensor::CovTapeEntry e{
                ex.root, rc,
                std::vector<float>(static_cast<size_t>(rows) * size * rc)};
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < size; ++c) {
                    const float* src =
                        &ex.jac[(static_cast<size_t>(r) * cols + k * size + c) *
                                rc];
                    float* dst =
                        &e.jac[(static_cast<size_t>(r) * size + c) * rc];
                    std::copy(src, src + rc, dst);
                }
            }
            out->cov_tape.push_back(std::move(e));
        }
        register_taped(out);
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
//  scale: z = c * x (deterministic constant)
// =============================================================================
TensorPtr scale(const TensorPtr& x, float c) {
    check_fresh({x});
    int n = static_cast<int>(x->mu.size());
    std::vector<float> mu(n), var(n);
    for (int i = 0; i < n; ++i) {
        mu[i] = c * x->mu[i];
        var[i] = c * c * x->var[i];
    }

    auto out =
        std::make_shared<GaussianTensor>(mu, var, x->shape, autoname("scale"));
    out->parents = {x};
    tape_unary(x, std::vector<float>(n, c), out);  // J = c I
    out->backward_fn = [x, c](const std::vector<float>& d_mu,
                              const std::vector<float>& d_var) {
        // Local factor g = cov(x, cx)/var(x) = c, exactly.
        std::vector<float> pd_mu(d_mu.size()), pd_var(d_var.size());
        for (size_t i = 0; i < d_mu.size(); ++i) {
            pd_mu[i] = c * d_mu[i];
            pd_var[i] = c * c * d_var[i];
        }
        x->accumulate(pd_mu, pd_var);
    };
    return out;
}

// =============================================================================
//  gather: out[i] = x[indices[i]], or exact zero where indices[i] == -1
// =============================================================================
TensorPtr gather(const TensorPtr& x, const std::vector<int>& indices, int rows,
                 int cols, const std::string& name_in) {
    check_fresh({x});
    size_t n_out = static_cast<size_t>(rows) * cols;
    if (indices.size() != n_out) {
        throw std::invalid_argument("gather: indices size " +
                                    std::to_string(indices.size()) +
                                    " != rows*cols = " + std::to_string(n_out));
    }
    int n_src = static_cast<int>(x->mu.size());
    std::vector<float> mu(n_out, 0.0f), var(n_out, 0.0f);
    for (size_t i = 0; i < n_out; ++i) {
        int src = indices[i];
        if (src >= n_src) {
            throw std::out_of_range("gather: index " + std::to_string(src) +
                                    " out of range for source size " +
                                    std::to_string(n_src));
        }
        if (src >= 0) {
            mu[i] = x->mu[src];
            var[i] = x->var[src];
        }
    }

    std::string name = name_in.empty() ? autoname("gather") : name_in;
    auto out = std::make_shared<GaussianTensor>(
        mu, var, std::vector<int>{rows, cols}, name);
    out->parents = {x};
    out->backward_fn = [x, indices](const std::vector<float>& d_mu,
                                    const std::vector<float>& d_var) {
        // gain = var/var = 1: scatter-add innovations back to the source;
        // a source referenced by several outputs accumulates all of them.
        std::vector<float> full_mu(x->mu.size(), 0.0f);
        std::vector<float> full_var(x->var.size(), 0.0f);
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= 0) {
                full_mu[indices[i]] += d_mu[i];
                full_var[indices[i]] += d_var[i];
            }
        }
        x->accumulate(full_mu, full_var);
    };
    return out;
}

// =============================================================================
//  matmul: batched GMA product of two graph tensors (see header; the same
//  independence treatment as production query_key() in src/attention.cpp)
// =============================================================================
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b, int batch, int m,
                 int k, int n, bool transpose_b) {
    check_fresh({a, b});
    size_t a_size = static_cast<size_t>(batch) * m * k;
    size_t b_size = static_cast<size_t>(batch) * k * n;
    if (a->mu.size() != a_size || b->mu.size() != b_size) {
        throw std::invalid_argument(
            "matmul: expected a of size batch*m*k = " + std::to_string(a_size) +
            " and b of size batch*k*n = " + std::to_string(b_size) + ", got " +
            std::to_string(a->mu.size()) + " and " +
            std::to_string(b->mu.size()));
    }

    // Flat index of b's element (t, j) within batch matrix bi.
    auto b_at = [batch, k, n, transpose_b](int bi, int t, int j) {
        return transpose_b ? (bi * n + j) * k + t : (bi * k + t) * n + j;
    };

    std::vector<float> mu(static_cast<size_t>(batch) * m * n, 0.0f);
    std::vector<float> var(static_cast<size_t>(batch) * m * n, 0.0f);
    for (int bi = 0; bi < batch; ++bi) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum_mu = 0.0f, sum_var = 0.0f;
                for (int t = 0; t < k; ++t) {
                    float ma = a->mu[(bi * m + i) * k + t];
                    float va = a->var[(bi * m + i) * k + t];
                    float mb = b->mu[b_at(bi, t, j)];
                    float vb = b->var[b_at(bi, t, j)];
                    sum_mu += ma * mb;
                    sum_var += va * vb + va * mb * mb + vb * ma * ma;
                }
                mu[(bi * m + i) * n + j] = sum_mu;
                var[(bi * m + i) * n + j] = sum_var;
            }
        }
    }

    auto out = std::make_shared<GaussianTensor>(
        mu, var, std::vector<int>{batch * m, n}, autoname("matmul"));
    out->parents = {a, b};
    out->backward_fn = [a, b, batch, m, k, n, b_at](
                           const std::vector<float>& d_mu,
                           const std::vector<float>& d_var) {
        // Local factors on normalized deltas: g(a_it -> z_ij) =
        // cov/var(a) = mu_b_tj and symmetrically g(b_tj -> z_ij) =
        // mu_a_it; a_it feeds every z_ij over j, b_tj over i.
        std::vector<float> ad_mu(a->mu.size(), 0.0f);
        std::vector<float> ad_var(a->var.size(), 0.0f);
        std::vector<float> bd_mu(b->mu.size(), 0.0f);
        std::vector<float> bd_var(b->var.size(), 0.0f);
        for (int bi = 0; bi < batch; ++bi) {
            for (int i = 0; i < m; ++i) {
                for (int t = 0; t < k; ++t) {
                    int ia = (bi * m + i) * k + t;
                    float ma = a->mu[ia];
                    float sum_a_mu = 0.0f, sum_a_var = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        int iz = (bi * m + i) * n + j;
                        int ib = b_at(bi, t, j);
                        float mb = b->mu[ib];
                        sum_a_mu += mb * d_mu[iz];
                        sum_a_var += mb * mb * d_var[iz];
                        bd_mu[ib] += ma * d_mu[iz];
                        bd_var[ib] += ma * ma * d_var[iz];
                    }
                    ad_mu[ia] = sum_a_mu;
                    ad_var[ia] = sum_a_var;
                }
            }
        }
        a->accumulate(ad_mu, ad_var);
        b->accumulate(bd_mu, bd_var);
    };
    return out;
}

// =============================================================================
//  softmax: row-wise, reusing production softmax_mean_var (diagonal jcb)
// =============================================================================
TensorPtr softmax(const TensorPtr& x) {
    check_fresh({x});
    int rows = x->shape[0];
    int cols = x->shape[1];
    int n = rows * cols;
    std::vector<float> mu(n), var(n), jcb(n);
    ::softmax_mean_var(x->mu, x->var, cols, rows, mu, jcb, var);

    auto out = std::make_shared<GaussianTensor>(mu, var, x->shape,
                                                autoname("softmax"));
    out->parents = {x};
    // Local factor g = cov(z,a)/var(z) = jcb on normalized deltas.
    out->backward_fn = [x, jcb](const std::vector<float>& d_mu,
                                const std::vector<float>& d_var) {
        std::vector<float> pd_mu(d_mu.size()), pd_var(d_var.size());
        for (size_t i = 0; i < d_mu.size(); ++i) {
            pd_mu[i] = jcb[i] * d_mu[i];
            pd_var[i] = jcb[i] * jcb[i] * d_var[i];
        }
        x->accumulate(pd_mu, pd_var);
    };
    return out;
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
    tape_linear(x, w->mu, batch, n_in, n_out, out);  // J_z = mu_W J_x

    out->backward_fn = [x, w, b, batch, n_in, n_out](
                           const std::vector<float>& d_mu,
                           const std::vector<float>& d_var) {
        const std::vector<float>& mu_x = x->mu;
        const std::vector<float>& mu_w = w->mu;

        // ---- normalized deltas for the input hidden states -- the same
        // multiply-accumulate as production linear_bwd_fc_delta_z:
        //   d_mu_x_k = sum_o mu_w[o,k] d_mu_o   (local factor g = mu_w)
        std::vector<float> xd_mu(batch * n_in, 0.0f);
        std::vector<float> xd_var(batch * n_in, 0.0f);
        for (int r = 0; r < batch; ++r) {
            for (int k = 0; k < n_in; ++k) {
                float sum_mu = 0.0f, sum_var = 0.0f;
                for (int o = 0; o < n_out; ++o) {
                    float mwok = mu_w[o * n_in + k];
                    sum_mu += d_mu[r * n_out + o] * mwok;
                    sum_var += d_var[r * n_out + o] * mwok * mwok;
                }
                xd_mu[r * n_in + k] = sum_mu;
                xd_var[r * n_in + k] = sum_var;
            }
        }
        x->accumulate(xd_mu, xd_var);

        // ---- normalized deltas for the parameters (local factor
        // g = mu_x), SUMMED over the batch like production
        // linear_bwd_fc_delta_w: each sample is an independent observation
        // of the same weights, so their updates accumulate; averaging
        // would shrink learning by a factor of batch_size. The var_w
        // multiplication happens once, in Parameter::apply_update. ----
        std::vector<float> dw_mu(n_out * n_in, 0.0f);
        std::vector<float> dw_var(n_out * n_in, 0.0f);
        for (int o = 0; o < n_out; ++o) {
            for (int k = 0; k < n_in; ++k) {
                float sum_mu = 0.0f, sum_var = 0.0f;
                for (int r = 0; r < batch; ++r) {
                    float mxk = mu_x[r * n_in + k];
                    sum_mu += d_mu[r * n_out + o] * mxk;
                    sum_var += d_var[r * n_out + o] * mxk * mxk;
                }
                dw_mu[o * n_in + k] = sum_mu;
                dw_var[o * n_in + k] = sum_var;
            }
        }
        w->accumulate(dw_mu, dw_var);

        if (b != nullptr) {
            // Bias local factor g = 1.
            std::vector<float> db_mu(n_out, 0.0f), db_var(n_out, 0.0f);
            for (int o = 0; o < n_out; ++o) {
                for (int r = 0; r < batch; ++r) {
                    db_mu[o] += d_mu[r * n_out + o];
                    db_var[o] += d_var[r * n_out + o];
                }
            }
            b->accumulate(db_mu, db_var);
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

}  // namespace tagi_autocov
