"""Attention + LSTM time series forecasting with tagi_autocov -- the TAGI
counterpart of examples/attention_lstm_torch.py.

Same data (SineDataset), same architecture (PositionalEncoding ->
MultiheadAttention -> LSTM -> LSTM -> Linear head; RMSNorm is omitted, it
has no tagi_autocov counterpart) -- but no optimizer, no loss, no
backward() code: training is one call, pred.observe(y, sigma_v**2), which
runs TAGI's analytical Bayesian update through the whole graph, attention
included. Unlike the torch model, predictions come with a predictive
variance, plotted as +/- 2 sigma bands.

The --parity flag runs a direct comparison of the two attention
implementations: torch.nn.MultiheadAttention's weights are copied into
tagi_autocov.MultiheadAttention (via Parameter.set_moments) and the
forward means are compared on the same input -- they agree to float32
precision, since with deterministic inputs TAGI's forward means reduce to
the standard computation.

Usage (needs an env with torch only for --parity):
    python -m examples.attention_lstm_tagi_autocov
    python -m examples.attention_lstm_tagi_autocov --parity
    python -m examples.attention_lstm_tagi_autocov --corrupt_steps '[5,15]'
"""

# Temporary import path setup. It will be removed in the final version.
import glob
import os
import sys

# Order matters: repo root must precede build/lib.* (which may hold a stale
# pytagi copy from a previous `setup.py build`), and build/lib.* must
# precede site-packages so the freshly built cutagi wins.
_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
for _p in glob.glob(os.path.join(_root, "build", "lib.*")) + [_root]:
    sys.path.insert(0, _p)

import cutagi
import fire
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pytagi.tagi_autocov import (
    LSTM,
    Linear,
    Module,
    MultiheadAttention,
    gather,
    rms_norm_rows,
)


class SineDataset:
    """Sine waves at different speeds -- same generator as the torch
    example, numpy only. The model sees seq_len steps, predicts the next;
    80/20 time-ordered train/test split."""

    def __init__(
        self,
        n_features: int = 6,
        seq_len: int = 20,
        n_samples: int = 4000,
        corrupt_steps: list = None,
    ):
        t = np.linspace(-40, 40, n_samples)
        data = np.sin(
            np.stack(
                [t / (np.pi / k) for k in range(1, n_features + 1)], axis=1
            )
        )
        x = np.stack(
            [data[i : i + seq_len] for i in range(n_samples - seq_len)]
        )
        y = data[seq_len:]

        self.corrupt_steps = corrupt_steps or []
        for s in self.corrupt_steps:
            x[:, s, :] = np.random.randn(len(x), n_features) * 5.0

        split = int(len(x) * 0.8)
        self.x_train, self.y_train = x[:split], y[:split]
        self.x_test, self.y_test = x[split:], y[split:]

    def next_batch(self, batch_size: int):
        idx = np.random.randint(len(self.x_train), size=batch_size)
        return self.x_train[idx], self.y_train[idx]


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal PE, identical to the torch example's buffer."""
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None].astype(float)
    div = np.exp(
        np.arange(0, d_model, 2).astype(float) * (-np.log(10000.0) / d_model)
    )
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class AttentionLSTM(Module):
    """PositionalEncoding -> MultiheadAttention -> RMSNorm -> LSTM -> LSTM
    -> Linear, mirroring the torch example's AttentionLSTM. Two Gaussian-
    specific stabilizers replace what a small SGD learning rate does for
    the torch model:

    * qk_norm=True inside the attention: without it, every observe()
      pushes the Q and K projections in aligned directions (scores are
      bilinear in both), the logits grow, the softmax saturates, and
      training diverges.
    * rms_norm_rows on the attention output (the torch model's RMSNorm
      slot): normalizes the Gaussian second moment mu^2 + var, since the
      attention block's output VARIANCE is what otherwise detonates the
      LSTM recurrence (var_f * mu_c^2 feedback)."""

    def __init__(
        self,
        embed_dim,
        out_size,
        num_heads=1,
        hidden_size=16,
        track_cross_cov: bool = True,
    ):
        self.embed_dim = embed_dim
        self.att = MultiheadAttention(embed_dim, num_heads, qk_norm=True)
        self.lstm1 = LSTM(
            embed_dim, hidden_size, track_cross_cov=track_cross_cov
        )
        self.lstm2 = LSTM(
            hidden_size, hidden_size, track_cross_cov=track_cross_cov
        )
        self.fc = Linear(hidden_size, out_size)
        self._step_idx_cache = {}

    def _step_indices(self, batch: int, seq: int):
        """Exact reindex {B*S, E} -> seq tensors of {B, E}: step t's
        element (b, e) is attention-output element (b*S + t)*E + e."""
        key = (batch, seq)
        if key not in self._step_idx_cache:
            E = self.embed_dim
            b = np.arange(batch)[:, None]
            e = np.arange(E)[None, :]
            self._step_idx_cache[key] = [
                ((b * seq + t) * E + e).flatten().tolist() for t in range(seq)
            ]
        return self._step_idx_cache[key]

    def forward(self, x: np.ndarray):
        batch, seq, E = x.shape
        att_out, attn = self.att(x + positional_encoding(seq, E))
        att_out = rms_norm_rows(att_out)
        steps = [
            gather(att_out, idx, batch, E)
            for idx in self._step_indices(batch, seq)
        ]
        out1, _ = self.lstm1(steps)
        out2, _ = self.lstm2(out1)
        return self.fc(out2[-1]), attn


def parity_check(
    embed_dim: int = 6, num_heads: int = 2, batch: int = 3, seq: int = 10
):
    """Copy torch.nn.MultiheadAttention weights into
    tagi_autocov.MultiheadAttention and compare forward outputs on the
    same input. With a deterministic input, tagi's forward MEANS follow
    the same equations as torch, so outputs must agree to float32
    precision. (Variances have no torch counterpart to compare.)"""
    import torch

    torch.manual_seed(0)
    t_mha = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    g_mha = MultiheadAttention(embed_dim, num_heads)

    E = embed_dim
    Wq, Wk, Wv = t_mha.in_proj_weight.detach().numpy().reshape(3, E, E)
    bq, bk, bv = t_mha.in_proj_bias.detach().numpy().reshape(3, E)
    Wo = t_mha.out_proj.weight.detach().numpy()
    bo = t_mha.out_proj.bias.detach().numpy()
    for lin, W, b in [
        (g_mha.q_proj, Wq, bq),
        (g_mha.k_proj, Wk, bk),
        (g_mha.v_proj, Wv, bv),
        (g_mha.out_proj, Wo, bo),
    ]:
        lin.weight.set_moments(W.flatten().tolist(), [0.0] * W.size)
        lin.bias.set_moments(b.tolist(), [0.0] * b.size)

    x = np.random.default_rng(1).normal(0, 1, (batch, seq, E))
    with torch.no_grad():
        t_out, t_attn = t_mha(
            *[torch.tensor(x, dtype=torch.float32)] * 3,
            need_weights=True,
            average_attn_weights=False,
        )
    g_out, g_attn = g_mha(x)

    out_err = np.abs(
        np.array(g_out.mu).reshape(batch, seq, E) - t_out.numpy()
    ).max()
    H = num_heads
    attn_err = np.abs(
        np.array(g_attn.mu).reshape(batch, H, seq, seq) - t_attn.numpy()
    ).max()
    print(
        "parity vs torch.nn.MultiheadAttention "
        f"(E={E}, H={num_heads}, B={batch}, S={seq}):"
    )
    print(f"  max |output diff|         = {out_err:.3e}")
    print(f"  max |attention-map diff|  = {attn_err:.3e}")
    ok = out_err < 1e-5 and attn_err < 1e-5
    print("  PARITY OK" if ok else "  PARITY FAILED")
    return ok


def main(
    num_epochs: int = 50,
    batch_size: int = 32,
    seq_len: int = 20,
    n_features: int = 6,
    num_heads: int = 1,
    hidden_size: int = 16,
    sigma_v: float = 4,
    sigma_v_min: float = 0.3,
    decay: float = 0.95,
    steps_per_epoch: int = 25,
    corrupt_steps: list = None,
    parity: bool = False,
    seed: int = 42,
):
    if parity:
        parity_check(embed_dim=n_features)
        return

    np.random.seed(seed)
    cutagi.manual_seed(seed)
    dataset = SineDataset(
        n_features=n_features, seq_len=seq_len, corrupt_steps=corrupt_steps
    )
    if dataset.corrupt_steps:
        print(f"Corrupted steps: {dataset.corrupt_steps} (in all samples)")
        print("Expect LOW attention weights at these columns.")

    model = AttentionLSTM(n_features, n_features, num_heads, hidden_size)

    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        mses = []
        for _ in range(steps_per_epoch):
            x, y = dataset.next_batch(batch_size)
            pred, _ = model(x)
            mses.append(np.mean((np.array(pred.mu) - y.flatten()) ** 2))
            pred.observe(y.flatten().tolist(), sigma_v**2)
        sigma_v = max(sigma_v_min, sigma_v * decay)
        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | mse: {np.mean(mses):.4f} "
            f"| sigma_v: {sigma_v:.2f}"
        )

    # Testing: batched forward over the test horizon, with uncertainty
    preds, pred_vars, targets = [], [], []
    for i in range(0, len(dataset.x_test) - batch_size + 1, batch_size):
        pred, _ = model(dataset.x_test[i : i + batch_size])
        preds.append(np.array(pred.mu).reshape(batch_size, n_features))
        pred_vars.append(np.array(pred.var).reshape(batch_size, n_features))
        targets.append(dataset.y_test[i : i + batch_size])
    preds = np.concatenate(preds)
    pred_std = np.sqrt(np.concatenate(pred_vars) + sigma_v**2)
    targets = np.concatenate(targets)
    test_mse = np.mean((preds - targets) ** 2)
    coverage = np.mean(np.abs(preds - targets) <= 2 * pred_std)
    print(f"\nTest MSE: {test_mse:.4f}")
    print(f"2-sigma coverage: {coverage * 100:.1f}% (expect ~95%)")

    os.makedirs("saved_results", exist_ok=True)
    t = np.arange(len(preds))
    ncols = 2
    nrows = (n_features + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.ravel()
    for f in range(n_features):
        ax = axes[f]
        ax.plot(t, targets[:, f], "k", lw=1, label="Actual")
        ax.plot(t, preds[:, f], "r", lw=1, label="Prediction")
        ax.fill_between(
            t,
            preds[:, f] - 2 * pred_std[:, f],
            preds[:, f] + 2 * pred_std[:, f],
            color="red",
            alpha=0.2,
            label="+/- 2 sigma",
        )
        ax.set_title(f"Feature {f}")
        ax.set_xlabel("Time step")
    axes[0].legend()
    fig.suptitle("TAGI-autocov Attention+LSTM Forecast", fontsize=14)
    plt.tight_layout()
    fig.savefig(
        "saved_results/attention_lstm_tagi_autocov.png", bbox_inches="tight"
    )
    plt.show()

    # Attention map of the first test sample (means)
    _, attn = model(dataset.x_test[:1])
    attn_mu = np.array(attn.mu).reshape(num_heads, seq_len, seq_len)
    print("\nAttention scores (head 0, query row 0):")
    print(f"  mu: {np.round(attn_mu[0, 0], 4)}")
    fig, ax = plt.subplots(
        1, num_heads, figsize=(4 * num_heads, 4), squeeze=False
    )
    for h in range(num_heads):
        im = ax[0][h].imshow(attn_mu[h], origin="lower", vmin=0)
        ax[0][h].set_xlabel("Key (time step)")
        ax[0][h].set_ylabel("Query (time step)")
        ax[0][h].set_title(f"Head {h + 1}")
        fig.colorbar(im, ax=ax[0][h])
        for s in dataset.corrupt_steps:
            ax[0][h].axvline(
                x=s, color="red", linestyle="--", lw=1.5, alpha=0.7
            )
    fig.savefig(
        "saved_results/attention_map_tagi_autocov.png", bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    fire.Fire(main)
