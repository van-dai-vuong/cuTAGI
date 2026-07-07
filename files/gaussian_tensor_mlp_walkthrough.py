"""
gaussian_tensor_mlp_walkthrough.py
==================================
GaussianTensor mechanics on a full (scalar) 1-hidden-layer network:

    x --> z1 = w1*x + b1 --> a = tanh(z1) --> z2 = w2*a + b2 --> observe y

The BACKWARD section below re-implements, BY HAND, exactly what the engine's
closures do -- the stepwise chain -- and compares every intermediate against
the engine.  No composite/telescoped covariance is used to compute anything;
the one-shot formula appears only at the very end as a cross-check that the
chain telescopes to it.

The engine's backward is three kinds of code:

    SEED   observe(y):        put (d_mu, d_var) on z2          (Eq. 5)
    HOPS   node._backward:    closure per op; input = the node's own packet,
                              output = parent._accumulate(g*d_mu, g^2*d_var)
                              with g = cov(parent, node)/var(node),
                              numerator FROZEN at forward time
    SINK   Parameter._apply_update:   mu += d_mu, var += d_var

We emulate each closure with a small function of the same shape.
"""

import numpy as np
from tagi_autograd import Parameter, linear, tanh, tensor

LINE = "-" * 74

# ---------------------------------------------------------------- priors
mu_x = 0.8
x = tensor([mu_x], var=0.0, name="x")  # deterministic
w1 = Parameter(np.array([[0.6]]), np.array([[0.10]]), name="w1")
b1 = Parameter(np.array([0.2]), np.array([0.05]), name="b1")
w2 = Parameter(np.array([[1.2]]), np.array([[0.08]]), name="w2")
b2 = Parameter(np.array([0.0]), np.array([0.04]), name="b2")

print(LINE)
print("PRIORS   x=0.8 (det.),  w1~N(0.6,0.10)  b1~N(0.2,0.05)")
print("                        w2~N(1.2,0.08)  b2~N(0.0,0.04)")

# ------------------------------------------------------- FORWARD, 3 hops
# Each op stores its LOCAL cross-covariances -- these are the only
# quantities the backward closures will ever read.
z1 = linear(x, w1, b1).retain()
cov_x_z1 = 0.0 * 0.6  # var_x * mu_w1  (var_x = 0)
cov_w1_z1 = 0.10 * mu_x  # var_w1 * mu_x
cov_b1_z1 = 0.05  # var_b1
print(LINE)
print("FORWARD hop 1   z1 = w1*x + b1")
print(f"  mu_z1={z1.mu[0]:.4f}  var_z1={z1.var[0]:.4f}")
print(
    f"  frozen: cov(w1,z1)={cov_w1_z1:.4f}  cov(b1,z1)={cov_b1_z1:.4f}"
    f"  cov(x,z1)={cov_x_z1:.4f}"
)

a = tanh(z1).retain()
J = 1 - np.tanh(z1.mu[0]) ** 2
cov_z1_a = J * z1.var[0]  # J * var_z1
print(LINE)
print("FORWARD hop 2   a = tanh(z1)")
print(f"  mu_a={a.mu[0]:.4f}  J={J:.4f}  var_a={a.var[0]:.4f}")
print(f"  frozen: cov(z1,a)={cov_z1_a:.4f}")

z2 = linear(a, w2, b2)
mu_a_, var_a_ = a.mu[0], a.var[0]
cov_a_z2 = var_a_ * 1.2  # var_a * mu_w2
cov_w2_z2 = 0.08 * mu_a_  # var_w2 * mu_a
cov_b2_z2 = 0.04  # var_b2
print(LINE)
print("FORWARD hop 3   z2 = w2*a + b2")
print(f"  mu_z2={z2.mu[0]:.4f}  var_z2={z2.var[0]:.4f}")
print(
    f"  frozen: cov(a,z2)={cov_a_z2:.4f}  cov(w2,z2)={cov_w2_z2:.4f}"
    f"  cov(b2,z2)={cov_b2_z2:.4f}"
)

# keep prior copies for the hand updates
mu_z1_p, var_z1_p = z1.mu[0], z1.var[0]
mu_z2_p, var_z2_p = z2.mu[0], z2.var[0]
w1_p, vw1_p = w1.mu[0, 0], w1.var[0, 0]
b1_p, vb1_p = b1.mu[0], b1.var[0]
w2_p, vw2_p = w2.mu[0, 0], w2.var[0, 0]
b2_p, vb2_p = b2.mu[0], b2.var[0]

# ======================================================================
# ENGINE: one call does the whole chain
# ======================================================================
y, var_v = 1.5, 0.05
z2.observe([y], var_v=var_v)

# ======================================================================
# HAND: replay the SAME chain, closure by closure (no composite formulas)
# ======================================================================


def hand_seed(y, mu_z, var_z, var_v):
    """what observe() does: Gaussian conditional against y = z + v."""
    var_y = var_z + var_v
    return var_z / var_y * (y - mu_z), -(var_z**2) / var_y


def hand_hop(cov_parent_node, var_node, d_mu, d_var):
    """one closure hop: parent gets (g*d_mu, g^2*d_var), g = cov/var."""
    g = cov_parent_node / var_node
    return g, g * d_mu, g**2 * d_var


print(LINE)
print("BACKWARD -- hand-executed chain, mirroring the engine's closures")

# ---- SEED on z2 ----
d_mu_z2, d_var_z2 = hand_seed(y, mu_z2_p, var_z2_p, var_v)
print(f"[SEED]  z2 packet: d_mu={d_mu_z2:+.4f}  d_var={d_var_z2:+.4f}")

# ---- z2's closure (linear2): fan out to a, w2, b2 -- ONE invocation ----
g, d_mu_a, d_var_a = hand_hop(cov_a_z2, var_z2_p, d_mu_z2, d_var_z2)
print(f"[z2._backward]  in: ({d_mu_z2:+.4f}, {d_var_z2:+.4f})")
print(
    f"   -> a   g={g:.4f}   packet ({d_mu_a:+.4f}, {d_var_a:+.4f})"
    f"   engine ({a.post_mu[0]-a.mu[0]:+.4f}, {a.post_var[0]-a.var[0]:+.4f})"
)
g, d_mu_w2, d_var_w2 = hand_hop(cov_w2_z2, var_z2_p, d_mu_z2, d_var_z2)
print(f"   -> w2  g={g:.4f}   packet ({d_mu_w2:+.4f}, {d_var_w2:+.4f})")
g, d_mu_b2, d_var_b2 = hand_hop(cov_b2_z2, var_z2_p, d_mu_z2, d_var_z2)
print(f"   -> b2  g={g:.4f}   packet ({d_mu_b2:+.4f}, {d_var_b2:+.4f})")

# ---- SINK for the output-layer parameters ----
print(
    f"[SINK]  w2: N({w2_p + d_mu_w2:.4f}, {vw2_p + d_var_w2:.4f})"
    f"   engine N({w2.mu[0,0]:.4f}, {w2.var[0,0]:.4f})"
)
print(
    f"        b2: N({b2_p + d_mu_b2:.4f}, {vb2_p + d_var_b2:.4f})"
    f"   engine N({b2.mu[0]:.4f}, {b2.var[0]:.4f})"
)

# ---- a's closure (tanh): single parent z1 ----
g, d_mu_z1, d_var_z1 = hand_hop(cov_z1_a, var_a_, d_mu_a, d_var_a)
print(f"[a._backward]   in: ({d_mu_a:+.4f}, {d_var_a:+.4f})")
print(
    f"   -> z1  g={g:.4f} (=1/J)   packet ({d_mu_z1:+.4f}, {d_var_z1:+.4f})"
    f"   engine ({z1.post_mu[0]-mu_z1_p:+.4f}, {z1.post_var[0]-var_z1_p:+.4f})"
)

# ---- z1's closure (linear1): fan out to x, w1, b1 ----
g, d_mu_x, d_var_x = hand_hop(cov_x_z1, var_z1_p, d_mu_z1, d_var_z1)
print(f"[z1._backward]  in: ({d_mu_z1:+.4f}, {d_var_z1:+.4f})")
print(f"   -> x   g={g:.4f}   packet ({d_mu_x:+.4f}, {d_var_x:+.4f})  (dies)")
g, d_mu_w1, d_var_w1 = hand_hop(cov_w1_z1, var_z1_p, d_mu_z1, d_var_z1)
print(f"   -> w1  g={g:.4f}   packet ({d_mu_w1:+.4f}, {d_var_w1:+.4f})")
g, d_mu_b1, d_var_b1 = hand_hop(cov_b1_z1, var_z1_p, d_mu_z1, d_var_z1)
print(f"   -> b1  g={g:.4f}   packet ({d_mu_b1:+.4f}, {d_var_b1:+.4f})")

# ---- SINK for the hidden-layer parameters ----
print(
    f"[SINK]  w1: N({w1_p + d_mu_w1:.4f}, {vw1_p + d_var_w1:.4f})"
    f"   engine N({w1.mu[0,0]:.4f}, {w1.var[0,0]:.4f})"
)
print(
    f"        b1: N({b1_p + d_mu_b1:.4f}, {vb1_p + d_var_b1:.4f})"
    f"   engine N({b1.mu[0]:.4f}, {b1.var[0]:.4f})"
)

# numeric assertions: the hand chain IS the engine
assert np.isclose(w2_p + d_mu_w2, w2.mu[0, 0])
assert np.isclose(vw2_p + d_var_w2, w2.var[0, 0])
assert np.isclose(w1_p + d_mu_w1, w1.mu[0, 0])
assert np.isclose(vw1_p + d_var_w1, w1.var[0, 0])
assert np.isclose(b1_p + d_mu_b1, b1.mu[0])
assert np.isclose(a.post_mu[0] - a.mu[0], d_mu_a)
assert np.isclose(z1.post_mu[0] - mu_z1_p, d_mu_z1)
print("all hand-chain values == engine values (asserted)")

# ======================================================================
# CROSS-CHECK ONLY: the chain telescopes to the one-shot Kalman formula.
# (The engine never computes this composite covariance -- it emerges.)
# ======================================================================
print(LINE)
var_y = var_z2_p + var_v
cov_w1_z2 = cov_w1_z1 / var_z1_p * cov_z1_a / var_a_ * cov_a_z2  # chained
print("TELESCOPING CROSS-CHECK (not used above):")
print(f"  chained cov(w1,z2) = cov(w1,z1)/var_z1 * cov(z1,a)/var_a * cov(a,z2)")
print(f"                     = {cov_w1_z2:.4f}")
print(f"  closed form var_w1*mu_x*J*mu_w2 = {0.10*mu_x*J*1.2:.4f}")
print(
    f"  one-shot d_mu_w1 = cov(w1,z2)/var_y*(y-mu_z2) "
    f"= {cov_w1_z2/var_y*(y-mu_z2_p):+.4f}   chain gave {d_mu_w1:+.4f}"
)
