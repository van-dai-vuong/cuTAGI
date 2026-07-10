"""
manual_parameter_update.py
==========================
The TAGI parameter update written BY HAND in plain NumPy -- no engine --
to show exactly what code performs the update.  Two versions:

  PART 1: scalar 1-hidden-layer net (same numbers as the walkthrough:
          w2 -> 1.3955, b2 -> 0.1653, w1 -> 0.8579, b1 -> 0.3612)

  PART 2: vectorized (n_out, n_in) layer -- the literal shapes/lines the
          engine uses -- checked against the engine to machine precision.

The recipe for ANY parameter theta feeding a node z that later reaches the
observed output:

    1. forward:  keep mu/var of every node, and the local covariances
                 cov(theta, z) = var_theta * mu_input      (weights)
                 cov(theta, z) = var_theta                 (biases)
    2. innovation at the output:  r = (y - mu_out) / (var_out + var_v)
    3. chain the innovation down to z through the local gains
                 gain(child -> parent) = cov(parent, child) / var(child)
       obtaining d_mu_z, d_var_z
    4. UPDATE (this is the whole "optimizer"):
                 g       = cov(theta, z) / var_z
                 mu_theta  += g   * d_mu_z
                 var_theta += g^2 * d_var_z          (d_var_z <= 0)
"""

import numpy as np

# =====================================================================
# PART 1 -- scalar net  x -> z1=w1*x+b1 -> a=tanh(z1) -> z2=w2*a+b2 -> y
# =====================================================================
print("=" * 70)
print("PART 1: scalar, by hand (compare with the walkthrough numbers)")

# ---- priors ----
mu_x = 0.8
mu_w1, var_w1 = 0.6, 0.10
mu_b1, var_b1 = 0.2, 0.05
mu_w2, var_w2 = 1.2, 0.08
mu_b2, var_b2 = 0.0, 0.04

# ---- 1. forward: moments + local covariances ----
mu_z1 = mu_w1 * mu_x + mu_b1
var_z1 = var_w1 * mu_x**2 + var_b1  # var_x = 0
cov_w1_z1 = var_w1 * mu_x  # <- weight covariance
cov_b1_z1 = var_b1  # <- bias covariance

mu_a = np.tanh(mu_z1)
J = 1 - np.tanh(mu_z1) ** 2
var_a = J**2 * var_z1
cov_z1_a = J * var_z1

mu_z2 = mu_w2 * mu_a + mu_b2
var_z2 = var_w2 * var_a + var_w2 * mu_a**2 + var_a * mu_w2**2 + var_b2
cov_a_z2 = var_a * mu_w2
cov_w2_z2 = var_w2 * mu_a
cov_b2_z2 = var_b2

# ---- 2. innovation at the observed output ----
y, var_v = 1.5, 0.05
var_y = var_z2 + var_v
d_mu_z2 = var_z2 / var_y * (y - mu_z2)  # Eq. 5
d_var_z2 = -(var_z2**2) / var_y

# ---- 4a. OUTPUT-LAYER PARAMETERS: one gain, then +=  ----
g_w2 = cov_w2_z2 / var_z2  # gain
mu_w2 += g_w2 * d_mu_z2  # <-- THE UPDATE
var_w2 += g_w2**2 * d_var_z2  # <--
g_b2 = cov_b2_z2 / var_z2
mu_b2 += g_b2 * d_mu_z2
var_b2 += g_b2**2 * d_var_z2
print(f"w2: N({mu_w2:.4f}, {var_w2:.4f})   b2: N({mu_b2:.4f}, {var_b2:.4f})")

# ---- 3. chain the innovation down to z1 ----
g_a = cov_a_z2 / var_z2  # z2 -> a
d_mu_a, d_var_a = g_a * d_mu_z2, g_a**2 * d_var_z2
g_z1 = cov_z1_a / var_a  # a -> z1   (= 1/J)
d_mu_z1, d_var_z1 = g_z1 * d_mu_a, g_z1**2 * d_var_a

# ---- 4b. HIDDEN-LAYER PARAMETERS: same two lines against z1 ----
g_w1 = cov_w1_z1 / var_z1
mu_w1 += g_w1 * d_mu_z1  # <-- THE UPDATE
var_w1 += g_w1**2 * d_var_z1  # <--
g_b1 = cov_b1_z1 / var_z1
mu_b1 += g_b1 * d_mu_z1
var_b1 += g_b1**2 * d_var_z1
print(f"w1: N({mu_w1:.4f}, {var_w1:.4f})   b1: N({mu_b1:.4f}, {var_b1:.4f})")
print("(walkthrough/engine gave w2=1.3955, b2=0.1653, w1=0.8579, b1=0.3612)")

# =====================================================================
# PART 2 -- vectorized layer z = W a + b, W:(n_out,n_in) -- the engine's
# actual lines, then verified against the engine.
# =====================================================================
print("=" * 70)
print("PART 2: vectorized (n_out, n_in), verified against the engine")

rng = np.random.default_rng(0)
n_in, n_out = 3, 2
mu_a_v = rng.normal(0, 1, n_in)  # Gaussian layer input
var_a_v = rng.uniform(0.05, 0.3, n_in)
mu_W = rng.normal(0, 0.5, (n_out, n_in))
var_W = rng.uniform(0.02, 0.1, (n_out, n_in))
mu_b = np.zeros(n_out)
var_b = np.full(n_out, 0.04)

# forward moments (GMA products summed + bias), diagonal covariances
mu_z = mu_W @ mu_a_v + mu_b
var_z = var_W @ var_a_v + var_W @ mu_a_v**2 + mu_W**2 @ var_a_v + var_b

# suppose the backward chain delivered these innovations on z
d_mu_z_v = np.array([0.30, -0.10])
d_var_z_v = np.array([-0.02, -0.01])

# ---------- the engine's parameter update, verbatim shapes ----------
r_mu = d_mu_z_v / var_z  # (n_out,)
r_var = d_var_z_v / var_z**2  # (n_out,)
#   cov(W[i,k], z_i) = var_W[i,k] * mu_a[k]   -> gain = cov / var_z[i]
d_mu_W = var_W * np.outer(r_mu, mu_a_v)  # (n_out, n_in)
d_var_W = var_W**2 * np.outer(r_var, mu_a_v**2)  # (n_out, n_in)
mu_W_post = mu_W + d_mu_W  # <-- THE UPDATE
var_W_post = var_W + d_var_W  # <--
mu_b_post = mu_b + var_b * r_mu  # cov(b_i, z_i) = var_b_i
var_b_post = var_b + var_b**2 * r_var

# ------------------------- engine cross-check ------------------------
from tagi_autocov import Parameter, linear, tensor

a_t = tensor(mu_a_v, var_a_v)
W_t = Parameter(mu_W.copy(), var_W.copy())
b_t = Parameter(mu_b.copy(), var_b.copy())
z_t = linear(a_t, W_t, b_t)
z_t._accumulate(d_mu_z_v, d_var_z_v)  # inject the same innovations
z_t.backward()

print("max |mu_W  hand - engine| =", np.abs(mu_W_post - W_t.mu).max())
print("max |var_W hand - engine| =", np.abs(var_W_post - W_t.var).max())
print("max |mu_b  hand - engine| =", np.abs(mu_b_post - b_t.mu).max())
print("max |var_b hand - engine| =", np.abs(var_b_post - b_t.var).max())
