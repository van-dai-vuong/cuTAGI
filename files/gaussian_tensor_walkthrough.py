"""
gaussian_tensor_walkthrough.py
==============================
How a GaussianTensor works, dissected on the smallest possible example:

        z = w * x + b,        observe  y = z + v

Every quantity is a scalar Gaussian so all numbers can be checked by hand.

A GaussianTensor is a graph node carrying:

    mu, var      prior moments  (diagonal covariance)
    parents      the inputs of the op that created it
    _backward    a CLOSURE stored at forward time; given THIS node's
                 accumulated innovations (d_mu, d_var), it pushes
                 gain-weighted innovations to each parent
    d_mu, d_var  accumulated innovations (the analogue of .grad)

The four stages below print the engine's numbers next to the manual
Kalman formulas.
"""

import numpy as np
from tagi_autocov import Parameter, add, linear, tensor

np.set_printoptions(precision=4, suppress=True)
LINE = "-" * 72

# =====================================================================
# STAGE 0 -- priors (chosen so every number below is clean)
# =====================================================================
x = tensor([2.0], var=0.0, name="x")  # deterministic input
w = Parameter(np.array([[0.5]]), np.array([[0.2]]), name="w")
b = Parameter(np.array([0.1]), np.array([0.1]), name="b")

print(LINE)
print("STAGE 0  priors")
print(f"  x ~ N(2.0, 0.0)   (deterministic covariate)")
print(f"  w ~ N(0.5, 0.2)")
print(f"  b ~ N(0.1, 0.1)")

# =====================================================================
# STAGE 1 -- forward: z = linear(x, w, b) builds ONE graph node.
#
# The op computes the prior moments of z AND freezes, inside the
# _backward closure, the cross-covariances of z with each parent:
#
#     mu_z  = mu_w mu_x + mu_b                    = 0.5*2 + 0.1 = 1.1
#     var_z = var_w mu_x^2 + var_x(...) + var_b   = 0.2*4 + 0.1 = 0.9
#
#     cov(w, z) = var_w * mu_x = 0.2*2 = 0.4      <- these three numbers
#     cov(b, z) = var_b        = 0.1              <- ARE the "gradient
#     cov(x, z) = var_x * mu_w = 0.0              <- tape" of this node
# =====================================================================
z = linear(x, w, b)
print(LINE)
print("STAGE 1  forward -> one GaussianTensor with a _backward closure")
print(f"  z.mu  = {z.mu[0]:.4f}      (hand: 1.1)")
print(f"  z.var = {z.var[0]:.4f}      (hand: 0.9)")
print(f"  z.parents = {[p.name for p in z.parents]}")
print(f"  stored cross-covariances (inside the closure):")
print(
    f"    cov(w,z) = var_w*mu_x = 0.4    cov(b,z) = var_b = 0.1"
    f"    cov(x,z) = 0"
)

# =====================================================================
# STAGE 2 -- observe(y):  y = z + v,  v ~ N(0, var_v)
#
# The Gaussian conditional (Eq. 5 of the paper) puts the FIRST innovation
# on z itself:
#
#     var_y  = var_z + var_v            = 0.9 + 0.1 = 1.0
#     d_mu_z  = var_z/var_y * (y-mu_z)  = 0.9 * 0.9  = 0.81
#     d_var_z = -var_z^2/var_y          = -0.81
#
# then triggers the backward sweep.
# =====================================================================
y, var_v = 2.0, 0.1
print(LINE)
print("STAGE 2  observe(y=2.0, var_v=0.1) -> innovation on z")
print(f"  var_y = {z.var[0] + var_v:.4f}")
print(f"  d_mu_z  = var_z/var_y*(y-mu_z) = 0.9*0.9 = 0.81")
print(f"  d_var_z = -var_z^2/var_y       = -0.81")

z.observe([y], var_v=var_v)  # runs stages 2-4 internally

# =====================================================================
# STAGE 3 -- z._backward fires (children before parents).
#
# The closure divides the innovations by z's prior variance once,
#
#     r_mu  = d_mu_z  / var_z    = 0.81 / 0.9  =  0.9
#     r_var = d_var_z / var_z^2  = -0.81/0.81  = -1.0
#
# and hands each parent   gain = cov(parent, z)/var_z   times innovation:
#
#     w: d_mu_w = cov(w,z)*r_mu = 0.4*0.9  = 0.36
#        d_var_w = cov(w,z)^2*r_var = 0.16*(-1) = -0.16
#     b: d_mu_b = 0.1*0.9 = 0.09,  d_var_b = 0.01*(-1) = -0.01
#     x: cov = 0 -> nothing flows into the deterministic input
#
# NOTE the composition: gain(z->w) * gain(y->z)
#      = cov(w,z)/var_z * var_z/var_y = cov(w,z)/var_y
# -- the intermediate var_z cancels, leaving exactly the one-shot Kalman
# gain of w against the OBSERVATION.  That cancellation is the engine's
# whole trick, repeated hop by hop in deep graphs.
# =====================================================================
print(LINE)
print("STAGE 3  z._backward pushed innovations to the parents (see above)")

# =====================================================================
# STAGE 4 -- Parameter leaves apply their accumulated innovations:
#     mu  <- mu  + d_mu,      var <- var + d_var
# and we can verify against the DIRECT joint-Gaussian conditional
#     mu_post  = mu  + cov(param, z)/var_y * (y - mu_z)
#     var_post = var - cov(param, z)^2/var_y
# =====================================================================
print(LINE)
print("STAGE 4  parameter posteriors (engine vs direct Kalman by hand)")
innov = (y - 1.1) / 1.0
print(
    f"  w: engine N({w.mu[0,0]:.4f}, {w.var[0,0]:.4f})   "
    f"hand N({0.5 + 0.4*innov:.4f}, {0.2 - 0.4**2/1.0:.4f})"
)
print(
    f"  b: engine N({b.mu[0]:.4f}, {b.var[0]:.4f})   "
    f"hand N({0.1 + 0.1*innov:.4f}, {0.1 - 0.1**2/1.0:.4f})"
)
print("  x: untouched (cov(x,z)=0: a var-0 leaf can absorb no information)")

# =====================================================================
# SECOND EXAMPLE -- fan-out: one node feeding TWO ops.
#
#     z1 = w1 * u,   z2 = w2 * u,   s = z1 + z2,   observe s
#
# u has two children, so during the sweep u ACCUMULATES one gain-weighted
# innovation per path (the covariance analogue of gradient accumulation):
#
#     d_mu_u = [var_u*mu_w1/var_z1 * var_z1/var_s
#             + var_u*mu_w2/var_z2 * var_z2/var_s] * d_mu_s
#            = var_u*(mu_w1 + mu_w2)/var_s * d_mu_s
#
# i.e. the sum over paths telescopes to cov(u, s)/var_s -- the correct
# one-shot gain.  (Caveat: var_s itself was formed under the usual TAGI
# independence assumption, ignoring that z1, z2 share u.)
# =====================================================================
print(LINE)
print("FAN-OUT  innovations from multiple children ACCUMULATE (sum)")
u = tensor([1.0], var=0.5, name="u").retain()
w1 = Parameter(np.array([[2.0]]), np.array([[0.0]]), name="w1")  # determin.
w2 = Parameter(np.array([[-1.0]]), np.array([[0.0]]), name="w2")
s = add(linear(u, w1), linear(u, w2))
mu_s, var_s = s.mu[0], s.var[0]
y2, var_v2 = 1.5, 0.1
s.observe([y2], var_v=var_v2)

d_mu_s = var_s / (var_s + var_v2) * (y2 - mu_s)
hand = 1.0 + 0.5 * (2.0 + (-1.0)) / var_s * d_mu_s  # sum over the 2 paths
print(
    f"  u posterior: engine {u.post_mu[0]:.6f}   hand (2-path sum) {hand:.6f}"
)
print(f"  = cov(u,s)/var_y * (y-mu_s) with cov(u,s) = var_u*(mu_w1+mu_w2)")
