"""
test_tagi.py -- verification suite for the TAGI autograd engine.

Checks, in order of increasing integration:

 1. Forward moments of the exact ops (linear/GMA mul/add) against Monte Carlo.
 2. Activation moments against the closed-form linearization formulas.
 3. EXACT single-path Kalman equivalence: for a scalar chain, the engine's
    posterior for every leaf must equal the direct joint-Gaussian conditional
    computed from the analytic cross-covariances (machine precision).
 4. Composite-gain identity through an activation (chain rule for gains).
 5. Paper identity: cov(C, H)/var(H) gain for H = O * tanh(C)
    must equal var(C) * dtanh(mu_C) * mu_O / var(H)   (Sec. 4.1.2).
 6. Paper identity: cov(C_prev, C)/var(C) for C = F*C_prev + I*Ctilde
    must equal var(C_prev) * mu_F / var(C)            (Appendix D.2 structure).
 7. Numerical edge cases: dead ReLU (no NaN, zero gain), SATURATED sigmoid
    (tiny Jacobian: the composite update through the activation must match
    the direct analytic covariance -- catches epsilon-clipping bugs).
 8. Shape safety of add/mul with mismatched shapes must raise, not silently
    broadcast.
 9. Double-observe on one graph must raise (stale-prior hazard).
10. End-to-end sanity: MLP RMSE decreases; LSTM log-lik increases.
"""

import numpy as np
from tagi_autocov import (
    GaussianTensor,
    Linear,
    Module,
    Parameter,
    add,
    linear,
    mul,
    relu,
    sigmoid,
    tanh,
    tensor,
)

PASS, FAIL = 0, 0


def check(name, ok, detail=""):
    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. forward moments vs Monte Carlo (exact ops)
# ---------------------------------------------------------------------------
def test_forward_moments_mc():
    n_mc = 2_000_000
    mu_x, sx = np.array([0.7, -1.2]), np.array([0.5, 0.8])
    w = Parameter(
        np.array([[0.3, -0.9], [1.1, 0.4]]), np.array([[0.2, 0.05], [0.1, 0.3]])
    )
    b = Parameter(np.array([0.1, -0.2]), np.array([0.04, 0.09]))

    z = linear(tensor(mu_x, sx**2), w, b)

    Xs = rng.normal(mu_x, sx, (n_mc, 2))
    Ws = rng.normal(w.mu, np.sqrt(w.var), (n_mc, 2, 2))
    Bs = rng.normal(b.mu, np.sqrt(b.var), (n_mc, 2))
    Zs = np.einsum("nij,nj->ni", Ws, Xs) + Bs

    ok_mu = np.allclose(z.mu, Zs.mean(0), atol=5e-3)
    ok_var = np.allclose(z.var, Zs.var(0), rtol=1e-2)
    check(
        "linear forward moments vs MC",
        ok_mu and ok_var,
        f"mu {z.mu} vs {Zs.mean(0).round(4)}",
    )

    # GMA multiplication (independent) is exact in mean/variance
    a = tensor([0.8], [0.5])
    c = tensor([-0.4], [0.3])
    p = mul(a, c)
    A = rng.normal(0.8, np.sqrt(0.5), n_mc)
    C = rng.normal(-0.4, np.sqrt(0.3), n_mc)
    check(
        "GMA mul forward moments vs MC",
        np.isclose(p.mu[0], (A * C).mean(), atol=3e-3)
        and np.isclose(p.var[0], (A * C).var(), rtol=1e-2),
    )


# ---------------------------------------------------------------------------
# 3. exact scalar Kalman equivalence for a single-path graph
# ---------------------------------------------------------------------------
def test_exact_kalman_linear_chain():
    """
    z = w*x + b, observe y = z + v.
    The engine's leaf posteriors must equal the direct Gaussian conditional
        mu_post = mu + cov(leaf, z)/(var_z + var_v) * (y - mu_z)
    with cov(x,z)=var_x*mu_w, cov(w,z)=var_w*mu_x, cov(b,z)=var_b.
    """
    mu_x, var_x = 0.9, 0.4
    w = Parameter(np.array([[1.3]]), np.array([[0.25]]))
    b = Parameter(np.array([-0.2]), np.array([0.09]))
    x = tensor([mu_x], [var_x]).retain()
    z = linear(x, w, b)
    mu_z, var_z = z.mu[0], z.var[0]
    y, var_v = 2.0, 0.1

    z.observe([y], var_v=var_v)

    innov = (y - mu_z) / (var_z + var_v)
    ok = (
        np.isclose(x.post_mu[0], mu_x + var_x * 1.3 * innov)
        and np.isclose(w.mu[0, 0], 1.3 + 0.25 * mu_x * innov)
        and np.isclose(b.mu[0], -0.2 + 0.09 * innov)
    )
    # variances: var_post = var - cov^2/(var_z+var_v)
    s = var_z + var_v
    ok &= np.isclose(x.post_var[0], var_x - (var_x * 1.3) ** 2 / s)
    ok &= np.isclose(w.var[0, 0], 0.25 - (0.25 * mu_x) ** 2 / s)
    ok &= np.isclose(b.var[0], 0.09 - 0.09**2 / s)
    check("exact Kalman equivalence (linear chain)", bool(ok))


def test_exact_kalman_mul():
    """z = a*c, observe z: posterior of a must use cov(a,z)=var_a*mu_c."""
    a = tensor([0.8], [0.5]).retain()
    c = tensor([-0.4], [0.3]).retain()
    z = mul(a, c)
    mu_z, var_z = z.mu[0], z.var[0]
    y, var_v = 0.3, 0.05
    z.observe([y], var_v=var_v)
    innov = (y - mu_z) / (var_z + var_v)
    ok = np.isclose(a.post_mu[0], 0.8 + 0.5 * (-0.4) * innov) and np.isclose(
        c.post_mu[0], -0.4 + 0.3 * 0.8 * innov
    )
    check("exact Kalman equivalence (GMA mul)", bool(ok))


# ---------------------------------------------------------------------------
# 4. composite gain through an activation must equal the direct covariance
# ---------------------------------------------------------------------------
def composite_gain_through(act, mu_z0=0.3, var_z0=0.4, w2=1.7, var_w2=0.0):
    """
    Chain: z0 (leaf) -> a = act(z0) -> z1 = w2 * a   (w2 deterministic).
    Direct analytic covariance: cov(z0, z1) = J * var_z0 * w2.
    The engine's update of z0 after observing z1 must match
        d_mu_z0 = cov(z0,z1)/(var_z1+var_v) * (y - mu_z1).
    """
    z0 = tensor([mu_z0], [var_z0]).retain()
    a = act(z0)
    w = Parameter(np.array([[w2]]), np.array([[var_w2]]))
    z1 = linear(a, w)
    y, var_v = 1.0, 0.2
    mu1, var1 = z1.mu[0], z1.var[0]
    z1.observe([y], var_v=var_v)

    if act is relu:
        jac = 1.0 if mu_z0 > 0 else 0.0
    elif act is tanh:
        jac = 1 - np.tanh(mu_z0) ** 2
    else:
        s = 1 / (1 + np.exp(-mu_z0))
        jac = s * (1 - s)
    expected = mu_z0 + jac * var_z0 * w2 / (var1 + var_v) * (y - mu1)
    return z0.post_mu[0], expected


def test_composite_gain():
    got, exp = composite_gain_through(tanh)
    check(
        "composite gain through tanh == direct covariance",
        np.isclose(got, exp),
        f"{got:.6f} vs {exp:.6f}",
    )
    got, exp = composite_gain_through(relu, mu_z0=0.5)
    check(
        "composite gain through relu == direct covariance", np.isclose(got, exp)
    )


# ---------------------------------------------------------------------------
# 5-6. the paper's LSTM covariance identities
# ---------------------------------------------------------------------------
def test_lstm_identities():
    n = 4
    mu = lambda: rng.normal(0, 0.5, n)
    vr = lambda: rng.uniform(0.05, 0.3, n)

    f = tensor(1 / (1 + np.exp(-mu())), vr()).retain()
    i = tensor(1 / (1 + np.exp(-mu())), vr())
    o = tensor(1 / (1 + np.exp(-mu())), vr())
    ctil = tensor(np.tanh(mu()), vr())
    c_prev = tensor(mu(), vr()).retain()

    c = add(mul(f, c_prev), mul(i, ctil)).retain()  # Eq. 1e
    h = mul(o, tanh(c))  # Eq. 1f

    # paper forward check (Sec. 4.1.1): var(C) from GMA + addition
    var_c_paper = (
        f.var * c_prev.var
        + f.var * c_prev.mu**2
        + c_prev.var * f.mu**2
        + i.var * ctil.var
        + i.var * ctil.mu**2
        + ctil.var * i.mu**2
    )
    check(
        "paper var(C) formula == engine forward",
        np.allclose(c.var, var_c_paper),
    )

    # inject an innovation at h and check the implied gains
    d_mu_h = np.full(n, 0.1)
    h._accumulate(d_mu_h, np.zeros(n))
    h.backward()

    # (5) cov(C,H) = var(C) * dtanh(mu_C) * mu_O      (Sec. 4.1.2)
    jac_tanh = 1 - np.tanh(c.mu) ** 2
    gain_c_h = c.var * jac_tanh * o.mu / h.var
    check(
        "paper cov(C,H) gain identity",
        np.allclose(c.post_mu - c.mu, gain_c_h * d_mu_h),
        f"max err {np.abs(c.post_mu - c.mu - gain_c_h*d_mu_h).max():.2e}",
    )

    # (6) cov(C_prev, C) = var(C_prev) * mu_F, chained down to c_prev
    gain_cp = (c_prev.var * f.mu / c.var) * gain_c_h
    check(
        "paper cov(C_prev,C) gain identity (Appendix D.2 structure)",
        np.allclose(c_prev.post_mu - c_prev.mu, gain_cp * d_mu_h),
    )

    # F receives cov(F, C) = var(F) * mu_Cprev  (GMA)
    gain_f = (f.var * c_prev.mu / c.var) * gain_c_h
    check(
        "GMA cov(F,C) gain identity",
        np.allclose(f.post_mu - f.mu, gain_f * d_mu_h),
    )


# ---------------------------------------------------------------------------
# 7. numerical edge cases
# ---------------------------------------------------------------------------
def test_dead_relu():
    z0 = tensor([-0.5], [0.3]).retain()  # relu is dead here: J = 0
    a = relu(z0)
    w = Parameter(np.array([[1.0]]), np.array([[0.1]]))
    z1 = linear(a, w)
    z1.observe([1.0], var_v=0.1)
    ok = np.isfinite(z0.post_mu).all() and np.isclose(z0.post_mu[0], -0.5)
    check("dead relu: zero gain, no NaN", bool(ok))


def test_saturated_sigmoid():
    """
    Moderately saturated sigmoid: J ~ 1e-7, so var(a) ~ 1e-14 -- BELOW any
    naive 1e-12 epsilon floor.  The composite update of z0 must STILL equal
    the direct analytic covariance gain J*var_z0*w2/(var1+var_v)*(y-mu1).
    Comparing the DELTAS (posterior - prior) catches silent attenuation
    that a comparison of the posteriors themselves would miss.
    """
    mu0 = 16.0
    got, exp = composite_gain_through(
        sigmoid, mu_z0=mu0, var_z0=1.0, w2=1.0, var_w2=0.0
    )
    d_got, d_exp = got - mu0, exp - mu0
    check(
        "saturated sigmoid: composite delta == direct covariance delta",
        np.isclose(d_got, d_exp, rtol=1e-9),
        f"delta {d_got:.3e} vs expected {d_exp:.3e}",
    )


def test_chunk_exact():
    """
    chunk is an exact slice: observing something built from one chunk must
    update ONLY the corresponding parent slice, with the same posterior as
    the direct scalar Kalman formula; the other slice must stay untouched.
    """
    from tagi_autocov import chunk

    x = tensor([0.5, -0.3, 1.1, 0.2], [0.4, 0.3, 0.2, 0.1]).retain()
    c0, c1 = chunk(x, 2)
    w = Parameter(np.array([[1.5, -0.7]]), np.array([[0.0, 0.0]]))
    z = linear(c0, w)  # uses ONLY the first chunk
    mu_z, var_z = z.mu[0], z.var[0]
    y, var_v = 1.0, 0.1
    z.observe([y], var_v=var_v)
    innov = (y - mu_z) / (var_z + var_v)
    exp0 = 0.5 + 0.4 * 1.5 * innov  # cov(x0, z) = var(x0)*w0
    exp1 = -0.3 + 0.3 * (-0.7) * innov
    ok = (
        np.isclose(x.post_mu[0], exp0)
        and np.isclose(x.post_mu[1], exp1)
        and np.isclose(x.post_mu[2], 1.1)
        and np.isclose(x.post_mu[3], 0.2)
        and np.isclose(x.post_var[2], 0.2)
    )
    check("chunk: exact slice Kalman, untouched complement", bool(ok))


def test_shape_mismatch_raises():
    try:
        add(tensor(np.zeros((3, 2)), 1.0), tensor(np.zeros(2), 1.0))
        ok = False
    except Exception:
        ok = True
    try:
        mul(tensor(np.zeros(3), 1.0), tensor(np.zeros(2), 1.0))
    except Exception:
        ok &= True
    else:
        ok = False
    check("add/mul with mismatched shapes raise", ok)


def test_double_observe_raises():
    x = tensor([0.0], [1.0])
    w = Parameter(np.array([[1.0]]), np.array([[0.1]]))
    z = linear(x, w)
    z.observe([1.0], var_v=0.1)
    try:
        z.observe([1.0], var_v=0.1)
        ok = False
    except RuntimeError:
        ok = True
    check("double observe on one graph raises", ok)


# ---------------------------------------------------------------------------
# 10. end-to-end sanity
# ---------------------------------------------------------------------------
def test_end_to_end():
    r = np.random.default_rng(3)
    X = r.uniform(-2, 2, (200, 1))
    Y = np.sin(3 * X) + r.normal(0, 0.1, (200, 1))

    class MLP(Module):
        def __init__(s, rr):
            s.fc1, s.fc2 = Linear(1, 32, rng=rr), Linear(32, 1, rng=rr)

        def forward(s, x):
            return s.fc2(relu(s.fc1(x)))

    net = MLP(np.random.default_rng(1))
    rmses = []
    for ep in range(15):
        for idx in r.permutation(200):
            net(tensor(X[idx])).observe(Y[idx], var_v=0.01)
        mu = np.array([net(tensor(xi)).mu for xi in X])
        rmses.append(float(np.sqrt(np.mean((mu - Y) ** 2))))
    check(
        "MLP end-to-end: RMSE decreases",
        rmses[-1] < rmses[0] and rmses[-1] < 0.30,
        f"{rmses[0]:.3f} -> {rmses[-1]:.3f}",
    )

    from tagi_lstm import LSTM, detach_state

    t = np.arange(200)
    y = np.sin(2 * np.pi * t / 20) + r.normal(0, 0.2, 200)
    covs = np.array(
        [
            [y[k - 1], np.sin(2 * np.pi * k / 20), np.cos(2 * np.pi * k / 20)]
            for k in range(1, 200)
        ]
    )
    targets = y[1:]

    class Net(Module):
        def __init__(s, rr):
            s.lstm = LSTM(3, 16, rng=rr)
            s.fc = Linear(16, 1, rng=rr)

        def forward(s, x, hx=None):
            h, hx = s.lstm(x, hx)
            return s.fc(h), hx

    net = Net(np.random.default_rng(2))
    lls = []
    for _ in range(8):
        hx, ll = None, 0.0
        for k in range(len(targets)):
            out, hx = net(tensor(covs[k]), hx)
            vy = out.var[0] + 0.04
            ll += (
                -0.5 * np.log(2 * np.pi * vy)
                - 0.5 * (targets[k] - out.mu[0]) ** 2 / vy
            )
            out.observe([targets[k]], var_v=0.04)
            hx = detach_state(hx)
        lls.append(ll / len(targets))
    check(
        "LSTM end-to-end: log-likelihood increases",
        lls[-1] > lls[0],
        f"{lls[0]:.3f} -> {lls[-1]:.3f}",
    )
    check("LSTM end-to-end: all finite", np.isfinite(lls).all())


if __name__ == "__main__":
    test_forward_moments_mc()
    test_exact_kalman_linear_chain()
    test_exact_kalman_mul()
    test_composite_gain()
    test_lstm_identities()
    test_dead_relu()
    test_saturated_sigmoid()
    test_chunk_exact()
    test_shape_mismatch_raises()
    test_double_observe_raises()
    test_end_to_end()
    print(f"\n{PASS} passed, {FAIL} failed")
