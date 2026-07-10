"""Inspecting tagi_autocov's covariance tape from Python.

The tape is what lets add()/mul() consider cross-covariances between nodes
that share stochastic ancestry (e.g. an LSTM's i_t and c_tilde_t, both
functions of [x_t, h_{t-1}]) automatically, at any graph depth:

  * ``node.track_cov()``    -- mark a node as a stochastic root; every op
                               downstream then carries the linearized
                               Jacobian d(node)/d(root) forward.
  * ``node.cov_tape``       -- debug view: one entry per root, with
                               jac[i][k] = d node[i] / d root[k].
  * ``cross_cov(a, b)``     -- the covariance matrix cov(a_i, b_j)
                               reconstructed from the roots BOTH nodes
                               depend on: shape (cols_a, cols_b) for
                               single-row nodes (e.g. cov(x, z) for
                               z = linear(x), x size 3, z size 5 is 3x5),
                               or (batch, cols_a, cols_b) for batched
                               nodes -- one block per sample, since batch
                               rows are independent. add()/mul() consume
                               each block's diagonal internally. An empty
                               list means "no shared root" = independent.
  * ``clear_cov_tapes()``   -- free all tapes (the LSTM calls this at the
                               start of every time step).

Run from the repository root:  python -m examples.tagi_autocov_tape_demo
"""

# Temporary import path setup. It will be removed in the final version.
import glob
import os
import sys

_build = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "build")
)
sys.path.append(_build)
sys.path.extend(glob.glob(os.path.join(_build, "lib.*")))

import cutagi
import numpy as np

import pytagi.tagi_autocov as pta

ta = cutagi.tagi_autocov

np.set_printoptions(precision=4, suppress=True)


def show(node, label):
    print(f"\n{label}  (name={node.name!r})")
    if not node.cov_tape:
        print("   cov_tape: EMPTY -> treated as independent of everything")
        return
    for e in node.cov_tape:
        J = np.array(e["jac"])
        print(f"   d({label})/d({e['root']}) =\n{J}")


def main():
    # ------------------------------------------------------------------
    # 1. Tiny graph: two "gates" reading the same x. Watch the Jacobian
    #    build up hop by hop (chain rule with expected slopes).
    # ------------------------------------------------------------------
    W1 = ta.Parameter([0.5, -0.3, 0.8, 0.4], [0.0] * 4, [2, 2], "W1")
    W2 = ta.Parameter([1.0, 0.2, -0.5, 0.7], [0.0] * 4, [2, 2], "W2")

    var_x = [0.5, 0.25]
    x = ta.tensor([0.7, -0.2], [1, 2], var_x, "x")
    x.track_cov()  # x becomes a root: its own Jacobian is the identity
    show(x, "x")

    z1 = ta.linear(x, W1)  # J(z1) = mu_W1                (1 hop)
    show(z1, "z1 = W1 x")

    a = ta.sigmoid(z1)  # J(a) = diag(jcb) J(z1)          (2 hops)
    show(a, "a = sigmoid(z1)")

    b = ta.tanh(ta.linear(x, W2))  # the other branch, same root
    show(b, "b = tanh(W2 x)")

    # cross_cov returns the FULL matrix cov(a_i, b_j) -- also between
    # nodes of different sizes, e.g. the input and output of a 2->5
    # linear layer: cov(x, z) = diag(var_x) mu_W^T, shape (2, 5).
    W3 = ta.Parameter([0.1 * k for k in range(10)], [0.0] * 10, [5, 2], "W3")
    z3 = ta.linear(x, W3)
    C_xz = np.array(ta.cross_cov(x, z3))
    print("\ncov(x, z3=W3 x), shape", C_xz.shape, ":\n", np.round(C_xz, 5))
    W3m = np.array([0.1 * k for k in range(10)]).reshape(5, 2)
    print("by hand diag(var_x) W3^T:\n", np.round(np.diag(var_x) @ W3m.T, 5))

    # The cross-covariance mul(a, b) will use is the DIAGONAL of the
    # full matrix (elementwise pairing a_i with b_i):
    C_ab = np.array(ta.cross_cov(a, b))
    print("\ncross_cov(a, b) =\n", np.round(C_ab, 5))

    # Check by hand: cov(a, b) = J_a diag(var_x) J_b^T
    Ja = np.array(a.cov_tape[0]["jac"])
    Jb = np.array(b.cov_tape[0]["jac"])
    print(
        "by hand J_a diag(var_x) J_b^T =\n",
        np.round(Ja @ np.diag(var_x) @ Jb.T, 5),
    )

    p = ta.mul(a, b)
    print(
        "\nmul(a,b): mu =",
        np.round(p.mu, 5),
        "  (mu_a*mu_b =",
        np.round(np.array(a.mu) * np.array(b.mu), 5),
        "+ diag =",
        np.round(np.diag(C_ab), 5),
        ")",
    )
    show(p, "p = a*b")  # product rule: J(p) = mu_b J(a) + mu_a J(b)

    # ------------------------------------------------------------------
    # 2. One LSTM step: three roots (x_t, h_prev, c_prev). The gate math
    #    is replicated from LSTM.step() with named intermediates so each
    #    node can be inspected.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LSTM: same mechanism with three roots")
    cutagi.manual_seed(0)
    lstm = pta.LSTM(2, 3)
    x_t = pta.tensor([0.3, -1.0], [1, 2], [0.2, 0.2], "x_t")
    prev = pta.LSTMState(
        pta.tensor([0.1, 0.2, -0.1], [1, 3], [0.3, 0.3, 0.3], "h_prev"),
        pta.tensor([0.5, -0.2, 0.0], [1, 3], [0.4, 0.4, 0.4], "c_prev"),
    )

    pta.clear_cov_tapes()
    x_t.track_cov()
    prev.h.track_cov()
    prev.c.track_cov()
    f = pta.sigmoid(pta.add(lstm.w_fx(x_t), lstm.w_fh(prev.h)))
    i = pta.sigmoid(pta.add(lstm.w_ix(x_t), lstm.w_ih(prev.h)))
    g = pta.tanh(pta.add(lstm.w_cx(x_t), lstm.w_ch(prev.h)))
    o = pta.sigmoid(pta.add(lstm.w_ox(x_t), lstm.w_oh(prev.h)))
    c = pta.add(pta.mul(f, prev.c), pta.mul(i, g))
    tc = pta.tanh(c)

    print("\nroots each node knows about:")
    for name, node in [
        ("i", i),
        ("g", g),
        ("c_t", c),
        ("tanh(c_t)", tc),
        ("o", o),
    ]:
        print(f"   {name:10s} -> {[e['root'] for e in node.cov_tape]}")

    # Elementwise (diagonal) covariances -- the terms var(c_t)/var(h_t) use:
    print(
        "\ndiag cov(i, g)       =",
        np.round(np.diag(np.array(pta.cross_cov(i, g))), 5),
        " (the i * c_tilde term of var(c_t))",
    )
    print(
        "diag cov(o, tanh(c)) =",
        np.round(np.diag(np.array(pta.cross_cov(o, tc))), 5),
        " (the term of var(h_t))",
    )
    print(
        "cov(f, prev.c)       =",
        pta.cross_cov(f, prev.c),
        " (no shared root -> empty = independent)",
    )


if __name__ == "__main__":
    main()
