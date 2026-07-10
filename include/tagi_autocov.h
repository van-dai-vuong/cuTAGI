#pragma once
/*
 * tagi_autocov: "autograd for Bayesian inference" in the spirit of TAGI
 * (Goulet et al. 2021; TAGI-LSTM paper). The name: besides an automatic
 * backward pass, the engine automatically tracks cross-COVariances between
 * nodes that share stochastic ancestry (the covariance tape below).
 *
 * You define only the FORWARD pass (subclass Module, override forward());
 * the backward pass is automatic. Every op builds a node that remembers its
 * parents and a backward closure; GaussianTensor::observe() seeds an
 * innovation at the output and replays those closures in reverse
 * topological order, exactly like autograd replays grad_fn's.
 *
 * What flows backward are NORMALIZED innovations -- the production
 * DeltaStates convention (see linear_bwd_fc_delta_z / the "innovation
 * vector" note in Goulet's pyTAGI-backward seminar):
 *
 *     d_mu  = (mu_post - mu_prior) / var        ("delta_mu")
 *     d_var = (var_post - var_prior) / var^2    ("delta_var")
 *
 * Why: every TAGI cross-covariance factors as the PARENT's own variance
 * times a local linear term, cov(parent, child) = var(parent) * g (g is
 * jcb for activations, mu_W for linear, mu_of_the_other_factor for GMA
 * products). Dividing the RTS recursion by var(parent) cancels that
 * leading factor, so the chain rule becomes a pure multiply-accumulate:
 *
 *     d_mu_parent  += g   * d_mu_child
 *     d_var_parent += g^2 * d_var_child
 *
 * The entire backward pass therefore contains exactly ONE division -- the
 * seed at observe(), d_mu = (y - mu)/(var + var_v), d_var = -1/(var +
 * var_v) -- and the consumers undo the normalization by multiplying by
 * their own variance once at the end: Parameter::apply_update does
 * mu += var * d_mu, var += var^2 * d_var, and retain() forms posteriors
 * the same way. A numerical bonus: a mid-chain variance that underflows
 * (a saturated sigmoid with var ~ 1e-14) never appears in a denominator.
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
namespace tagi_autocov {

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

    // -----------------------------------------------------------------
    // Forward-mode covariance tape (automatic cross-covariance).
    //
    // Two nodes that share stochastic ancestry (e.g. an LSTM's input and
    // candidate gates, both functions of [x_t, h_{t-1}]) are NOT
    // independent, yet add()/mul() would treat them as such. The tape
    // fixes this automatically: mark the shared sources with track_cov()
    // and every op propagates the locally-linearized sensitivity
    // J = d(node)/d(root) alongside (mu, var), so a binary op can
    // reconstruct the cross-covariance of its two operands as
    //
    //     cov(a_i, b_i) = sum_r [J_a diag(var_r) J_b^T]_ii
    //
    // (weights/biases never enter: distinct parameters are independent,
    // so only the shared ROOTS generate cross-covariance -- the same
    // structure as the hand-derived lstm_cov_input_cell_states /
    // lstm_cov_output_tanh_cell_states in src/lstm_layer.cpp, which this
    // mechanism re-derives by chaining the ops' own Jacobians).
    //
    // Layout: one entry per root this node depends on; jac is flat
    // (rows * cols * root_cols), batch-row-aligned -- jac[(b*cols + c) *
    // root_cols + k] = d node[b,c] / d root[b,k]. Batch rows are
    // independent (true for every row-wise op), so cross-batch blocks
    // are structurally zero and not stored. Roots must have the same
    // number of rows as the nodes built from them.
    //
    // track_cov() RESETS the node's tape to {self: I}: the node becomes
    // an exogenous Gaussian source with its CURRENT marginal variance,
    // discarding sensitivities to earlier roots. This is exactly the
    // TAGI-LSTM treatment of h_{t-1}/c_{t-1} at each step, and it keeps
    // the tape's depth (and cost) bounded to one step of a rollout.
    //
    // Tapes cost memory (rows*cols*root_cols floats per node per root);
    // call clear_cov_tapes() at the start of each rollout step -- before
    // re-marking that step's roots -- so only one step's tapes are ever
    // alive. gather/matmul/softmax drop the tape (their output rows are
    // not batch-row-aligned with the roots); everything else propagates.
    // -----------------------------------------------------------------
    struct CovTapeEntry {
        const GaussianTensor* root;  // identity key; kept alive via parents
        int root_cols;               // root columns per batch row
        std::vector<float> jac;      // (rows * cols * root_cols)
    };
    std::vector<CovTapeEntry> cov_tape;

    std::vector<TensorPtr> parents;
    std::function<void(const std::vector<float>&, const std::vector<float>&)>
        backward_fn;

    // Accumulated NORMALIZED innovations (analogue of .grad; see the
    // intro comment): d_mu = (mu_post - mu_prior)/var, d_var =
    // (var_post - var_prior)/var^2. Valid iff has_innovation.
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

    // Mark this node as a stochastic root of the covariance tape (see the
    // block comment above cov_tape). Chainable, like retain().
    GaussianTensor& track_cov();

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

    // Reverse-topological sweep: chain local factors on the normalized
    // deltas, apply parameter updates.
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

// Free every covariance tape in the process (nodes are found through a
// weak registry, so nothing is kept alive). Call at the start of each
// rollout step, before track_cov()-ing that step's roots, so tape memory
// stays bounded to one step.
void clear_cov_tapes();

// Debug/introspection: the cross-covariance matrix
//     cov(x[b,i], y[b,j]) = sum_r [J_x diag(var_r) J_y^T]_ij
// reconstructed from the roots both tapes share, returned flat as one
// (cols_x, cols_y) block per batch row b -- shape (rows * cols_x * cols_y).
// Batch rows are independent throughout the engine, so covariance between
// different rows is structurally zero and not stored: for {16,8} nodes the
// result is 16 blocks of (8,8), not a mostly-zero (128,128) matrix. x and
// y may have different column counts: for z = linear(x) with x {1,3}
// tracked and z {1,5}, cov(x, z) is the single (3, 5) block
// diag(var_x) mu_W^T. Returns an empty vector when x and y share no
// tracked roots (i.e. they are treated as independent). add()/mul()
// internally consume only the DIAGONAL of each block, since they pair
// same-shape operands element by element.
std::vector<float> cross_cov(const TensorPtr& x, const TensorPtr& y);

// ---------------------------------------------------------------------------
// Elementary ops -- each one is the whole "op library" the engine needs;
// arbitrary graphs (skip connections, gating, ...) compose from these.
// ---------------------------------------------------------------------------

// z = x + y. Without shared tape roots (the default): independent,
// cov(x,z) = var(x). With shared roots, the tape supplies rho = cov(x,y)
// per element and the op is exact for the correlated pair:
//     var(z) = var(x) + var(y) + 2 rho,   cov(x,z) = var(x) + rho.
TensorPtr add(const TensorPtr& x, const TensorPtr& y);

// z = x * y (GMA). Without shared tape roots: independent factors,
// cov(x,z) = var(x) mu_y. With shared roots, uses the exact moments of a
// product of jointly Gaussian variables with cov(x,y) = rho:
//     mu(z)  = mu_x mu_y + rho
//     var(z) = var_x var_y + var_x mu_y^2 + var_y mu_x^2
//              + rho^2 + 2 rho mu_x mu_y
//     cov(x,z) = var_x mu_y + rho mu_x
// -- the same correlated-product treatment as the production LSTM's
// lstm_cell_state_mean_var / lstm_hidden_state_mean_var.
TensorPtr mul(const TensorPtr& x, const TensorPtr& y);

// a = phi(z): every activation below computes (mu_a, var_a, jcb) per
// element from (mu_z, var_z) using the exact same formulas as their
// namesake `*_mean_var` functions in src/activation.cpp (relu/sigmoid/tanh
// are the local-linearization case jcb = phi'(mu_z), var_a = jcb^2 var_z;
// the mixture_*/softplus/leaky_relu variants use closed-form truncated- or
// rectified-Gaussian moments). In every case cov(z,a) = jcb * var(z), so
// the backward local factor cov(z,a)/var(z) = jcb chains exactly like
// the simple case.
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

// z = c * x, exact deterministic scaling: cov(x,z) = c var(x), so the
// backward local factor on normalized deltas is exactly c. Used e.g.
// for attention's 1/sqrt(d_head).
TensorPtr scale(const TensorPtr& x, float c);

// Arbitrary reindexing: out[i] = x[indices[i]] (flat indices into x), or
// an exact zero (mu=0, var=0) where indices[i] == -1. Like chunk, an
// output element IS its source element, so the backward local factor is
// exactly 1
// and innovations scatter-ADD back (a source referenced by several output
// positions -- e.g. overlapping conv patches -- accumulates all of them).
// One op covers im2col, padding, permute/reshape, head split/merge, and
// max-pooling (indices chosen by argmax of mu at forward time, the same
// selection rule as the production MaxPool2d). Drops the covariance tape
// (arbitrary reindexing breaks the batch-row alignment the tape assumes).
TensorPtr gather(const TensorPtr& x, const std::vector<int>& indices, int rows,
                 int cols, const std::string& name = "");

// Batched matrix product of two GRAPH tensors (both uncertain), assuming
// independence between and within a and b -- the same GMA treatment as the
// production attention's query_key() in src/attention.cpp:
//     mu_z  = sum_t mu_a mu_b
//     var_z = sum_t var_a var_b + var_a mu_b^2 + var_b mu_a^2
// a is `batch` stacked (m, k) matrices, flat {batch*m, k}. b is `batch`
// stacked (k, n) matrices, flat {batch*k, n} -- or (n, k) matrices, flat
// {batch*n, k}, when transpose_b (the Q K^T case). Innovations flow to
// BOTH parents with local factors cov(a,z)/var(a) = mu_b (and
// symmetrically mu_a for b) on the normalized deltas; unlike linear()
// there is no summing across the
// batch dimension -- each stacked matrix is a distinct state, not a
// shared parameter.
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b, int batch, int m,
                 int k, int n, bool transpose_b = false);

// Row-wise softmax over the last axis, reusing the production
// softmax_mean_var from src/activation.cpp: mu_a = softmax(mu_z) with
// max-subtraction, jcb = mu_a (1 - mu_a) (DIAGONAL approximation -- the
// off-diagonal -mu_i mu_j terms of the true softmax Jacobian are dropped,
// exactly as the production Softmax layer drops them), var_a = jcb^2
// var_z. Note the production MHA layer uses Remax instead, which tracks
// cross-covariances this engine's diagonal model cannot represent.
TensorPtr softmax(const TensorPtr& x);

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
// implemented in Python on top of this -- see pytagi/tagi_autocov.py)
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

}  // namespace tagi_autocov
