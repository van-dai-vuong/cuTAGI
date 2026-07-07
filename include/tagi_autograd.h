#pragma once
/*
 * tagi_autograd: "autograd for Bayesian inference" in the spirit of TAGI
 * (Goulet et al. 2021; TAGI-LSTM paper).
 *
 * You define only the FORWARD pass (subclass Module, override forward());
 * the backward pass is automatic. Every op builds a node that remembers its
 * parents and a backward closure; GaussianTensor::observe() seeds an
 * innovation at the output and replays those closures in reverse
 * topological order, exactly like autograd replays grad_fn's -- except
 * what is chained is not a Jacobian-vector product but the local TAGI gain
 *
 *     J(parent -> child) = cov(parent, child) / var(child)
 *
 * applied to the innovations d_mu = mu_post - mu_prior, d_var = var_post -
 * var_prior via the RTS-like recursion:
 *
 *     d_mu_parent  += J   * d_mu_child
 *     d_var_parent += J^2 * d_var_child
 *
 * Every GaussianTensor is stored as a flat row-major 2-D array with shape
 * {rows, cols}. For activations/hidden states, rows is the batch size
 * (rows == 1 for a single sample); for a weight matrix, rows/cols are
 * {n_out, n_in}. Treating the single-sample case as batch size 1 lets the
 * Linear parameter update (summed over the batch, matching the production
 * linear_bwd_fc_delta_w) apply uniformly, with no separate ndim==1 branch.
 */

#include <cmath>
#include <functional>
#include <memory>
#include <string>
#include <vector>

// Everything lives in this namespace: the library already has global
// `Linear`/`Sequential`/`Parameter` classes with unrelated (batched,
// hand-written-backward) semantics, so these names must not collide.
namespace tagi_autograd {

class GaussianTensor;
using TensorPtr = std::shared_ptr<GaussianTensor>;

void set_trace(bool enabled = true);

// A Gaussian random tensor N(mu, diag(var)) and a node in the graph.
class GaussianTensor : public std::enable_shared_from_this<GaussianTensor> {
   public:
    std::vector<float> mu;
    std::vector<float> var;
    std::vector<int> shape;  // {rows, cols}
    std::string name;

    std::vector<TensorPtr> parents;
    std::function<void(const std::vector<float>&, const std::vector<float>&)>
        backward_fn;

    // Accumulated innovations (analogue of .grad); valid iff has_innovation.
    std::vector<float> d_mu;
    std::vector<float> d_var;
    bool has_innovation = false;

    // Set once a backward sweep consumes this node (Parameters are exempt).
    bool used = false;

    // If true, the posterior moments (prior + innovation) survive the
    // backward sweep in post_mu/post_var -- used to carry recurrent state
    // (h_t|t, c_t|t) forward as the prior leaf of the next time step.
    bool retain_flag = false;
    bool has_post = false;
    std::vector<float> post_mu;
    std::vector<float> post_var;

    GaussianTensor(std::vector<float> mu, std::vector<float> var,
                   std::vector<int> shape, std::string name = "");
    virtual ~GaussianTensor() = default;

    GaussianTensor& named(const std::string& n);
    GaussianTensor& retain();

    // A fresh leaf carrying this node's posterior moments if a backward
    // sweep filled them, else its prior moments. Mirrors PyTorch's
    // `h.detach()` in truncated-BPTT loops.
    TensorPtr detach() const;

    void accumulate(const std::vector<float>& d_mu_in,
                    const std::vector<float>& d_var_in);

    // Condition on y = self + v, v ~ N(0, var_v), then propagate the
    // update back to every hidden state and parameter (Parameters are
    // updated in place).
    void observe(const std::vector<float>& y, float var_v = 0.0f);

    // Reverse-topological sweep: chain gains, apply parameter updates.
    void backward();
};

// A leaf Gaussian tensor whose (mu, var) persist and get updated.
class Parameter : public GaussianTensor {
   public:
    Parameter(std::vector<float> mu, std::vector<float> var,
              std::vector<int> shape, std::string name = "param");

    void apply_update();
};
using ParamPtr = std::shared_ptr<Parameter>;

// Wrap data (e.g. deterministic covariates: var=0) as a graph leaf.
TensorPtr tensor(std::vector<float> mu, std::vector<int> shape,
                 std::vector<float> var = {}, std::string name = "x");

// ---------------------------------------------------------------------------
// Elementary ops -- each one is the whole "op library" the engine needs;
// arbitrary graphs (skip connections, gating, ...) compose from these.
// ---------------------------------------------------------------------------

// z = x + y, independent: cov(x,z) = var(x).
TensorPtr add(const TensorPtr& x, const TensorPtr& y);

// z = x * y, independent factors (GMA): cov(x,z) = var(x) * mu_y.
TensorPtr mul(const TensorPtr& x, const TensorPtr& y);

// a = phi(z): every activation below computes (mu_a, var_a, jcb) per
// element from (mu_z, var_z) using the exact same formulas as their
// namesake `*_mean_var` functions in src/activation.cpp (relu/sigmoid/tanh
// are the local-linearization case jcb = phi'(mu_z), var_a = jcb^2 var_z;
// the mixture_*/softplus/leaky_relu variants use closed-form truncated- or
// rectified-Gaussian moments). In every case cov(z,a) = jcb * var(z), so
// the backward gain cov(z,a)/var(a) chains exactly like the simple case.
// softmax/remax/closed_form_softmax are NOT included: they normalize
// jointly across a whole row (need cov(a_i, a_j) for i != j), which this
// engine's diagonal (elementwise-independent) covariance model can't
// represent -- they'd need a dedicated non-diagonal op, not a fit for
// this activation_op abstraction.
TensorPtr relu(const TensorPtr& x);
TensorPtr tanh_act(const TensorPtr& x);
TensorPtr sigmoid(const TensorPtr& x);
TensorPtr mixture_relu(const TensorPtr& x);
TensorPtr mixture_sigmoid(const TensorPtr& x);
TensorPtr mixture_tanh(const TensorPtr& x);
TensorPtr softplus(const TensorPtr& x);
TensorPtr leaky_relu(const TensorPtr& x, float alpha = 0.1f);

// Heteroscedastic-noise output transform: passes even-indexed elements
// through unchanged and applies A = exp(Z) (lognormal moments) to
// odd-indexed elements, matching src/activation.cpp's even_exp_mean_var
// (there used to turn interleaved [mean, log-var, mean, log-var, ...]
// outputs into [mean, var, mean, var, ...]).
TensorPtr even_exp(const TensorPtr& x);

// Split the last axis into `chunks` equal parts. Gain is exactly 1: a
// chunk IS a slice of its parent.
std::vector<TensorPtr> chunk(const TensorPtr& x, int chunks,
                             const std::string& name = "");

// Affine map z = x W^T + b (GMA product + addition fused), with the
// parameter cross-covariances cov(x_k,z_i) = var(x_k) mu_W[i,k],
// cov(W_ik,z_i) = var(W_ik) mu_x_k, cov(b_i,z_i) = var(b_i).
TensorPtr linear(const TensorPtr& x, const ParamPtr& w,
                 const ParamPtr& b = nullptr);

// ---------------------------------------------------------------------------
// torch-like Module layer on top. C++ has no __dict__ reflection, so a
// Module must explicitly register_parameter()/register_module() (same
// constraint libtorch's C++ API has); parameters() then aggregates
// recursively, mirroring Python's Module.parameters().
// ---------------------------------------------------------------------------
class Module {
   public:
    virtual ~Module() = default;

    virtual TensorPtr forward(const TensorPtr& x) = 0;
    TensorPtr operator()(const TensorPtr& x) { return forward(x); }

    virtual std::vector<ParamPtr> parameters() const;

   protected:
    void register_parameter(const ParamPtr& p);
    void register_module(const std::shared_ptr<Module>& m);

   private:
    std::vector<ParamPtr> params_;
    std::vector<std::shared_ptr<Module>> submodules_;
};

// Bayesian dense layer with a weakly informative Gaussian prior. Weight
// and bias moments come from src/param_init.cpp's init_weight_bias_linear
// (the same function the production Linear layer uses), drawing from the
// process-wide SeedManager -- call cutagi.manual_seed(seed) for
// reproducibility, same convention as the rest of the library.
//
// `bias` can be disabled, e.g. when two Linears are summed to emulate a
// single affine map over a concatenated input (as in an LSTM cell,
// implemented in Python on top of this -- see pytagi/tagi_autograd.py)
// and only one of the two should own the additive bias term.
//
// `fan_in_override`, when > 0, is used in place of n_in only for the
// He/Xavier scale computation (shape is still (n_out, n_in)). This lets
// an LSTM gate's two split Linears (input->hidden and hidden->hidden)
// each get the same scale a single fused (input_size+hidden_size)
// -> hidden_size matrix would have under init_weight_bias_lstm, by
// passing fan_in_override = input_size + hidden_size to both.
class Linear : public Module {
   public:
    ParamPtr weight;
    ParamPtr bias;  // nullptr when constructed with bias=false

    Linear(int n_in, int n_out, bool bias = true, float gain_w = 1.0f,
           float gain_b = 1.0f, const std::string& init_method = "He",
           int fan_in_override = 0);

    TensorPtr forward(const TensorPtr& x) override;
};

// Wraps a plain function (e.g. relu) as a parameter-free Module so it can
// sit inside a Sequential alongside Linear layers.
class FunctionModule : public Module {
   public:
    explicit FunctionModule(std::function<TensorPtr(const TensorPtr&)> fn);

    TensorPtr forward(const TensorPtr& x) override;

   private:
    std::function<TensorPtr(const TensorPtr&)> fn_;
};

std::shared_ptr<Module> relu_module();
std::shared_ptr<Module> tanh_module();
std::shared_ptr<Module> sigmoid_module();

class Sequential : public Module {
   public:
    std::vector<std::shared_ptr<Module>> items;

    explicit Sequential(std::vector<std::shared_ptr<Module>> items);

    TensorPtr forward(const TensorPtr& x) override;
};

}  // namespace tagi_autograd
