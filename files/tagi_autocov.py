"""
tagi_autocov.py
================
A standalone pure-Python "autograd for Bayesian inference" engine in the
spirit of TAGI (Goulet et al. 2021; Vuong et al., TAGI-LSTM paper), kept
as readable reference material. NOTE: unlike the C++ engine
(include/tagi_autocov.h), this mirror does NOT implement the covariance
tape (add/mul here treat their operands as independent), and it chains
RAW innovations with gains cov/var per op, whereas the C++ engine flows
variance-NORMALIZED deltas (single division at observe(), production
DeltaStates convention).

You define only the FORWARD pass with torch-like syntax:

    class MLP(Module):
        def __init__(self):
            self.fc1 = Linear(1, 64)
            self.fc2 = Linear(64, 1)
        def forward(self, x):
            return self.fc2(relu(self.fc1(x)))

    net = MLP()
    out = net(x)                    # forward: builds the graph and stores
                                    # cross-covariances node by node
    out.observe(y, var_v=0.01)      # backward: automatic layer-wise Gaussian
                                    # inference; parameters updated in place

WHAT REPLACES THE GRADIENT
--------------------------
In autograd, every op stores local derivatives and backward chains them by
multiplication.  Here every op stores the LOCAL GAIN

    J(parent -> child) = cov(parent, child) / var(child)

and the backward pass chains gains while propagating the *innovations*

    d_mu  = mu_post  - mu_prior
    d_var = var_post - var_prior

with the RTS-like recursion (Eq. 6 / Eq. 9 of the paper):

    d_mu_parent  += J   * d_mu_child
    d_var_parent += J^2 * d_var_child

THE FOUR LOCAL RULES -- every hop in every chain is one of these; the gain
is always cov(parent, node)/var(node) with the numerator frozen at forward
time (the "tape"):

    op                  gain to each parent
    ------------------  ------------------------------------------------
    GMA multiply u*v    to u: var(u)*mu_v / var(uv)
    addition u+v        to u: var(u) / var(u+v)
    activation phi(z)   to z: J*var(z)/var(phi) = 1/J   (J = phi'(mu_z))
    linear Wx+b         to W: var(W)*mu_x/var(z);  to b: var(b)/var(z);
                        to x: var(x)*mu_W/var(z)
    (chunk/slice        gain exactly 1 -- a chunk IS its parent's slice)

Debugging: call set_trace(True) to print the backward sweep as it executes
([SEED] -> [CLOSURE in/out] -> [SINK] lines with the actual packets).

The chain rule works because gains compose exactly like Jacobians:
cov(z, z'')/var(z'') factorizes through the intermediate node, so updating
through an activation node then a linear node reproduces the composite
covariance cov(Z^(j), Z^(j+1)) derived analytically in the paper.

The local cross-covariances come from the three elementary operations:

  1. ADDITION       z = x + y  (independent) : cov(x,z) = var(x)
  2. MULTIPLICATION z = x * y  (independent) : GMA ->  cov(x,z) = var(x) mu_y
                                               var(z) = vx vy + vx my^2 + vy mx^2
  3. ACTIVATION     a = phi(z) (linearized)  : cov(z,a) = J var(z),
                                               var(a)   = J^2 var(z),  J = phi'(mu_z)

All covariances are diagonal (TAGI independence assumptions); every tensor
carries (mu, var) with an optional leading batch dimension.

Author: generated with Claude
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12  # floor for POSTERIOR variances only (must stay > 0)

_TRACE = False
_OP_COUNTER: dict = {}


def set_trace(enabled: bool = True):
    """
    When enabled, the backward sweep prints itself:
      [SEED]    observe() creating the first packet
      [CLOSURE] each node's _backward invocation with its INPUT packet
      OUT ->    each parent._accumulate it performs (gain-weighted packet)
      [SINK]    each Parameter applying its accumulated innovations
    Zero overhead when disabled.
    """
    global _TRACE
    _TRACE = enabled


def _autoname(op: str) -> str:
    _OP_COUNTER[op] = _OP_COUNTER.get(op, 0) + 1
    return f"{op}{_OP_COUNTER[op]}"


def _fmt(v) -> str:
    v = np.asarray(v, dtype=float)
    if v.size == 1:
        return f"{float(v.ravel()[0]):+.4f}"
    return f"[{v.shape} max|.|={np.abs(v).max():.4f}]"


def _safe_div(a, b):
    """
    Exact elementwise a/b where b > 0, and 0 where b == 0.

    IMPORTANT: gain denominators must NOT be floored at some epsilon.
    A TAGI gain cov(parent, child)/var(child) is a Kalman gain -- the
    numerator always shrinks together with the denominator, so the exact
    ratio is bounded, while flooring the denominator silently attenuates
    updates through low-variance nodes (e.g. saturated sigmoids where
    var(a) = J^2 var(z) ~ 1e-14) by orders of magnitude.
    """
    b = np.asarray(b, dtype=float)
    pos = b > 0.0
    return np.where(pos, np.asarray(a, float) / np.where(pos, b, 1.0), 0.0)


# =============================================================================
#  Core node: a Gaussian tensor in the computational graph
# =============================================================================
class GaussianTensor:
    """
    A Gaussian random tensor N(mu, diag(var)).

    Graph bookkeeping mimics autograd:
        parents      : tuple of GaussianTensor inputs of the op that made it
        _backward    : closure that, given this node's accumulated innovations
                       (d_mu, d_var), pushes gain-weighted innovations to the
                       parents.  This is the analogue of grad_fn.
        d_mu, d_var  : accumulated innovations (analogue of .grad)
    """

    def __init__(self, mu, var, parents=(), backward_fn=None, name=""):
        self.mu = np.asarray(mu, dtype=float)
        self.var = np.asarray(var, dtype=float)
        if self.var.shape != self.mu.shape:
            self.var = np.broadcast_to(self.var, self.mu.shape).copy()
        self.parents = parents
        self._backward = backward_fn
        self.name = name
        self.d_mu = None
        self.d_var = None
        self._retain = False  # keep posterior moments after backward
        self._used = False  # set once a backward sweep consumed it
        self.post_mu = None  # mu_post  (filled if retained)
        self.post_var = None  # var_post (filled if retained)

    def detach(self):
        """
        Return a FRESH LEAF carrying this node's posterior moments if a
        backward sweep filled them (retained nodes after observe), else its
        prior moments.  Mirrors PyTorch's `h.detach()` in truncated-BPTT
        loops: the returned tensor has no parents, so the next forward pass
        starts a new graph from the filtered state h_t|t, c_t|t.
        """
        mu = self.post_mu if self.post_mu is not None else self.mu
        var = self.post_var if self.post_var is not None else self.var
        return GaussianTensor(
            np.array(mu, dtype=float),
            np.array(var, dtype=float),
            name=self.name + ".detached",
        )

    def named(self, name: str):
        """Rename this node (chainable) -- makes set_trace() output and
        reuse-guard errors readable, e.g. h = mul(o, tc).named("h")."""
        self.name = name
        return self

    def retain(self):
        """
        Mark this node so its POSTERIOR moments (prior + innovation) survive
        the backward sweep in .post_mu / .post_var.  Used for recurrent
        states: the posterior h_t|t, c_t|t of one time step becomes the
        prior leaf of the next (the filtering recursion of the paper).
        """
        self._retain = True
        return self

    # ------------------------------------------------------------------ utils
    @property
    def shape(self):
        return self.mu.shape

    def std(self):
        return np.sqrt(self.var)

    def __repr__(self):
        return (
            f"GaussianTensor(name={self.name!r}, shape={self.shape}, "
            f"mu~{self.mu.ravel()[:3]}, var~{self.var.ravel()[:3]})"
        )

    def _accumulate(self, d_mu, d_var):
        if _TRACE:
            print(
                f"      OUT -> {self.name}._accumulate("
                f"d_mu={_fmt(d_mu)}, d_var={_fmt(d_var)})"
            )
        if self.d_mu is None:
            self.d_mu = np.zeros_like(self.mu)
            self.d_var = np.zeros_like(self.var)
        self.d_mu += d_mu
        self.d_var += d_var

    # ------------------------------------------------- operator overloading
    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __matmul__(self, other):
        raise TypeError(
            "Use Linear(...) for affine maps so the engine can "
            "store the parameter cross-covariances."
        )

    # =====================================================================
    #  observe(): the whole point.  Output-layer Gaussian conditional (Eq.5)
    #  followed by an automatic backward sweep through the graph (Eq. 6).
    # =====================================================================
    def observe(self, y, var_v=0.0):
        """
        Condition the graph on the observation  y = self + v,
        v ~ N(0, var_v), then propagate the update back to every hidden
        state and parameter.  Parameters are updated IN PLACE.
        """
        if self._used:
            raise RuntimeError(
                "observe(): this graph was already conditioned; parameters "
                "were updated so its cached priors are stale. Re-run the "
                "forward pass to build a fresh graph."
            )
        y = np.asarray(y, dtype=float)
        if _TRACE:
            print(
                f"[SEED]    observe(y={_fmt(y)}, var_v={var_v}) "
                f"on {self.name!r}"
            )
        var_y = self.var + var_v  # predictive variance
        gain = _safe_div(self.var, var_y)  # cov(Y, Z^O) / var(Y)

        d_mu = gain * (y - self.mu)  # mu_post  - mu_prior
        d_var = -gain * self.var  # var_post - var_prior
        self._accumulate(d_mu, d_var)
        self.backward()

    def backward(self):
        """Reverse-topological sweep: chain gains, apply parameter updates."""
        topo, seen = [], set()

        def dfs(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for p in node.parents:
                dfs(p)
            topo.append(node)

        dfs(self)
        for node in reversed(topo):  # children before parents
            if not isinstance(node, Parameter):  # Parameters ARE reusable
                node._used = True  # everything else is consumed
            has_innov = node.d_mu is not None
            if node._retain:  # save posterior = prior+innov
                node.post_mu = node.mu + (node.d_mu if has_innov else 0.0)
                node.post_var = np.maximum(
                    node.var + (node.d_var if has_innov else 0.0), _EPS
                )
            if not has_innov:
                continue
            if node._backward is not None:
                if _TRACE:
                    print(
                        f"[CLOSURE] {node.name}._backward  IN: "
                        f"d_mu={_fmt(node.d_mu)}, d_var={_fmt(node.d_var)}"
                    )
                node._backward(node.d_mu, node.d_var)
            if isinstance(node, Parameter):
                node._apply_update()
            # free innovations (graph is per-forward, like autograd)
            node.d_mu = node.d_var = None


class Parameter(GaussianTensor):
    """A leaf Gaussian tensor whose (mu, var) persist and get updated."""

    def __init__(self, mu, var, name="param"):
        super().__init__(mu, var, parents=(), backward_fn=None, name=name)

    def _apply_update(self):
        if _TRACE:
            print(
                f"[SINK]    {self.name}: mu {_fmt(self.mu)} -> "
                f"{_fmt(self.mu + self.d_mu)}, var {_fmt(self.var)} -> "
                f"{_fmt(np.maximum(self.var + self.d_var, _EPS))}"
            )
        self.mu = self.mu + self.d_mu
        self.var = np.maximum(self.var + self.d_var, _EPS)


def _check_fresh(*nodes):
    """Ops must not extend a graph whose priors were already conditioned."""
    for n in nodes:
        if getattr(n, "_used", False):
            raise RuntimeError(
                f"node {n.name!r} belongs to an already-conditioned graph; "
                "its stored moments are stale priors. Re-run the forward pass."
            )


def tensor(mu, var=0.0, name="x"):
    """Wrap data (e.g. deterministic covariates: var=0) as a graph leaf."""
    mu = np.asarray(mu, dtype=float)
    return GaussianTensor(
        mu, np.broadcast_to(np.asarray(var, float), mu.shape).copy(), name=name
    )


# =============================================================================
#  Op 1: ADDITION of independent Gaussians
# =============================================================================
def add(x: GaussianTensor, y: GaussianTensor) -> GaussianTensor:
    _check_fresh(x, y)
    if x.mu.shape != y.mu.shape:
        raise ValueError(
            f"add: shape mismatch {x.mu.shape} vs {y.mu.shape}; "
            "broadcast explicitly so innovations are well-defined"
        )
    mu = x.mu + y.mu
    var = x.var + y.var  # independence

    def backward_fn(d_mu, d_var, x=x, y=y, out_var=var):
        for p in (x, y):
            j = _safe_div(p.var, out_var)  # cov(p, z)/var(z) = var(p)/var(z)
            p._accumulate(j * d_mu, j**2 * d_var)

    return GaussianTensor(
        mu, var, parents=(x, y), backward_fn=backward_fn, name=_autoname("add")
    )


# =============================================================================
#  Op 2: element-wise MULTIPLICATION (GMA, independent factors)
# =============================================================================
def mul(x: GaussianTensor, y: GaussianTensor) -> GaussianTensor:
    _check_fresh(x, y)
    if x.mu.shape != y.mu.shape:
        raise ValueError(
            f"mul: shape mismatch {x.mu.shape} vs {y.mu.shape}; "
            "broadcast explicitly so innovations are well-defined"
        )
    if x is y:
        raise ValueError(
            "mul(x, x): GMA assumes independent factors; "
            "x*x needs the correlated-GMA terms (cov(X1,X2)!=0)"
        )
    mu = x.mu * y.mu
    var = x.var * y.var + x.var * y.mu**2 + y.var * x.mu**2

    def backward_fn(d_mu, d_var, x=x, y=y, out_var=var):
        # GMA cross-covariance: cov(x, xy) = var(x) mu_y  (cov(x,y)=0)
        jx = _safe_div(x.var * y.mu, out_var)
        jy = _safe_div(y.var * x.mu, out_var)
        x._accumulate(jx * d_mu, jx**2 * d_var)
        y._accumulate(jy * d_mu, jy**2 * d_var)

    return GaussianTensor(
        mu, var, parents=(x, y), backward_fn=backward_fn, name=_autoname("mul")
    )


# =============================================================================
#  Op 3: ACTIVATIONS via local linearization
# =============================================================================
def _activation(x: GaussianTensor, f, df, name) -> GaussianTensor:
    _check_fresh(x)
    jac = df(x.mu)  # J = phi'(mu_z)
    mu = f(x.mu)
    var = jac**2 * x.var
    cov_z_a = jac * x.var  # cov(Z, A) = J var(Z)

    def backward_fn(d_mu, d_var, x=x, cov=cov_z_a, out_var=var):
        j = _safe_div(cov, out_var)  # exact 1/J where var(a) > 0, else 0
        x._accumulate(j * d_mu, j**2 * d_var)
        check = 1

    return GaussianTensor(
        mu, var, parents=(x,), backward_fn=backward_fn, name=_autoname(name)
    )


def relu(x):
    return _activation(
        x, lambda m: np.maximum(0.0, m), lambda m: (m > 0).astype(float), "relu"
    )


def tanh(x):
    return _activation(x, np.tanh, lambda m: 1 - np.tanh(m) ** 2, "tanh")


def sigmoid(x):
    s = lambda m: 1.0 / (1.0 + np.exp(-m))
    return _activation(x, s, lambda m: s(m) * (1 - s(m)), "sigmoid")


# =============================================================================
#  chunk: split a tensor into equal parts along the last axis.
#  Each chunk IS a slice of its parent, so cov(parent_slice, chunk) =
#  var(chunk) and the backward gain is exactly 1: innovations scatter back
#  into the parent's slice positions unchanged.
# =============================================================================
def chunk(x: GaussianTensor, chunks: int, name: str | None = None):
    _check_fresh(x)
    name = name or _autoname("chunk")
    n = x.mu.shape[-1]
    if n % chunks:
        raise ValueError(f"chunk: last dim {n} not divisible by {chunks}")
    size = n // chunks
    outs = []
    for k in range(chunks):
        sl = slice(k * size, (k + 1) * size)

        def backward_fn(d_mu, d_var, x=x, sl=sl):
            full_mu = np.zeros_like(x.mu)
            full_var = np.zeros_like(x.var)
            full_mu[..., sl] = d_mu  # gain = var/var = 1
            full_var[..., sl] = d_var
            x._accumulate(full_mu, full_var)

        outs.append(
            GaussianTensor(
                x.mu[..., sl].copy(),
                x.var[..., sl].copy(),
                parents=(x,),
                backward_fn=backward_fn,
                name=f"{name}[{k}]",
            )
        )
    return outs


# =============================================================================
#  Affine map  z = x W^T + b  (multiplication + addition fused, with the
#  parameter cross-covariances of the paper's Appendix C-style derivations)
# =============================================================================
def linear(
    x: GaussianTensor, w: Parameter, b: Parameter | None = None
) -> GaussianTensor:
    """
    x : (..., n_in)      w : (n_out, n_in)      b : (n_out,) or None

    Forward moments (sum of independent GMA products + bias):
        mu_z  = mu_x mu_W^T + mu_b
        var_z = var_x var_W^T + mu_x^2 var_W^T + var_x (mu_W^2)^T + var_b

    Saved cross-covariances (diagonal):
        cov(x_k, z_i) = var(x_k) mu_W[i,k]
        cov(W_ik, z_i) = var(W_ik) mu_x_k
        cov(b_i,  z_i) = var(b_i)
    """
    _check_fresh(x)
    mu_x, var_x = x.mu, x.var
    mu = mu_x @ w.mu.T
    var = var_x @ w.var.T + (mu_x**2) @ w.var.T + var_x @ (w.mu**2).T
    if b is not None:
        mu = mu + b.mu
        var = var + b.var

    def backward_fn(d_mu, d_var, x=x, w=w, b=b, out_var=var):
        r_mu = _safe_div(d_mu, out_var)  # d_mu_i  / var(z_i)
        r_var = _safe_div(d_var, out_var**2)  # d_var_i / var(z_i)^2

        # ---- innovations for the input hidden states ----
        #   d_mu_x_k = sum_i cov(x_k,z_i)/var(z_i) d_mu_i
        x._accumulate((r_mu @ w.mu) * var_x, (r_var @ (w.mu**2)) * var_x**2)

        # ---- innovations for the parameters (averaged over batch) ----
        if mu_x.ndim == 1:
            nb = 1.0
            dw_mu = np.outer(r_mu, mu_x)
            dw_var = np.outer(r_var, mu_x**2)
            db_mu, db_var = r_mu, r_var
        else:
            nb = float(mu_x.shape[0])
            dw_mu = r_mu.T @ mu_x
            dw_var = r_var.T @ (mu_x**2)
            db_mu, db_var = r_mu.sum(0), r_var.sum(0)

        w._accumulate(w.var * dw_mu / nb, w.var**2 * dw_var / nb)
        if b is not None:
            b._accumulate(b.var * db_mu / nb, b.var**2 * db_var / nb)

    parents = (x, w) if b is None else (x, w, b)
    return GaussianTensor(
        mu,
        var,
        parents=parents,
        backward_fn=backward_fn,
        name=_autoname("linear"),
    )


# =============================================================================
#  torch-like Module layer on top
# =============================================================================
class Module:
    def parameters(self):
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Parameter):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, Module):
                        params.extend(u.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """Bayesian dense layer with a weakly informative Gaussian prior."""

    def __init__(self, n_in, n_out, gain=1.0, rng=None):
        rng = rng or np.random.default_rng()
        s = gain / np.sqrt(n_in)
        self.weight = Parameter(
            rng.normal(0, s, (n_out, n_in)),
            np.full((n_out, n_in), s**2),
            name="W",
        )
        self.bias = Parameter(np.zeros(n_out), np.full(n_out, s**2), name="b")

    def forward(self, x):
        return linear(x, self.weight, self.bias)


class Sequential(Module):
    def __init__(self, *modules_and_fns):
        self.items = modules_and_fns

    def parameters(self):
        params = []
        for it in self.items:
            if isinstance(it, Module):
                params.extend(it.parameters())
        return params

    def forward(self, x):
        for it in self.items:
            x = it(x)
        return x


# =============================================================================
#  Demo
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # ---------------------------------------------------------------
    # 1) The user only defines the forward pass -- 2-layer MLP
    # ---------------------------------------------------------------
    class MLP(Module):
        def __init__(self, rng):
            self.fc1 = Linear(1, 64, rng=rng)
            self.fc2 = Linear(64, 1, rng=rng)

        def forward(self, x):
            z1 = self.fc1(x)
            z2 = relu(z1)
            output = self.fc2(z2)
            return output

    def truth(x):
        return np.sin(3 * x) + 0.3 * x**2

    n = 300
    x_train = rng.uniform(-2, 2, size=(n, 1))
    y_train = truth(x_train) + rng.normal(0, 0.1, size=(n, 1))

    net = MLP(np.random.default_rng(1))
    sigma_v = 0.1

    for epoch in range(50):
        for i in rng.permutation(n):
            out = net(tensor(x_train[i]))  # forward builds the graph
            out.observe(y_train[i], var_v=sigma_v**2)  # auto backward
        if epoch % 10 == 0 or epoch == 49:
            mus = np.array([net(tensor(xi)).mu for xi in x_train])
            rmse = np.sqrt(np.mean((mus - y_train) ** 2))
            print(f"epoch {epoch:3d} | train RMSE = {rmse:.4f}")

    print("\npredictions with uncertainty:")
    for xv in [-1.5, 0.0, 0.6, 1.5, 2.4]:
        out = net(tensor([xv]))
        sd = np.sqrt(out.var[0] + sigma_v**2)
        print(
            f"  x={xv:5.2f}  truth={truth(xv):7.3f} "
            f" pred={out.mu[0]:7.3f} +/- {sd:.3f}"
        )

    # ---------------------------------------------------------------
    # 2) Arbitrary graphs work too (skip connection + element-wise mult):
    #    the backward inference follows whatever graph the forward built.
    # ---------------------------------------------------------------
    print("\narbitrary-graph example (residual * gate):")

    class GatedResNet(Module):
        def __init__(self, rng):
            self.inp = Linear(1, 32, rng=rng)
            self.h = Linear(32, 32, rng=rng)
            self.gate = Linear(32, 32, rng=rng)
            self.out = Linear(32, 1, rng=rng)

        def forward(self, x):
            a = relu(self.inp(x))
            branch = tanh(self.h(a)) * sigmoid(self.gate(a))  # GMA product
            return self.out(a + branch)  # skip add

    net2 = GatedResNet(np.random.default_rng(3))
    for epoch in range(50):
        for i in rng.permutation(n):
            net2(tensor(x_train[i])).observe(y_train[i], var_v=sigma_v**2)
    mus = np.array([net2(tensor(xi)).mu for xi in x_train])
    print(
        f"  GatedResNet train RMSE = "
        f"{np.sqrt(np.mean((mus - y_train) ** 2)):.4f}"
    )
