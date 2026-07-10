"""Torch-like Python layer on top of ``cutagi.tagi_autocov`` (the pybind11
binding of include/tagi_autocov.h). You define only the forward pass by
subclassing ``Module``; calling ``.observe(y, var_v)`` on the output runs
the TAGI backward sweep through the graph built by forward() and updates
every registered ``Parameter`` in place -- there is no backward()/step()
to write.

Weight/bias initialization for Linear (and, through it, LSTM's gates)
comes from src/param_init.cpp's ``init_weight_bias_linear`` -- the exact
function the production ``pytagi.nn.Linear``/``LSTM`` layers use -- drawn
from the process-wide ``SeedManager``. Call ``cutagi.manual_seed(seed)``
once for reproducibility, same convention as the rest of pytagi (there is
no per-Linear seed argument, matching production layers).

``LSTM`` here is one layer, same as torch.nn.LSTM -- stack instances for a
multi-layer network exactly like PyTorch:

    class Net(Module):
        def __init__(self, embed_dim, hidden_size, out_size):
            self.lstm1 = LSTM(embed_dim, hidden_size)
            self.lstm2 = LSTM(hidden_size, hidden_size)
            self.fc = Linear(hidden_size, out_size)

        def forward(self, x, hx=None):
            hx1, hx2 = hx if hx is not None else (None, None)
            out1, hx1 = self.lstm1(x, hx1)
            out2, hx2 = self.lstm2(out1, hx2)
            return self.fc(out2[-1]), (hx1, hx2)

The LSTM gate math is written here in Python from the tagi_autocov
primitives (Linear, add, mul, sigmoid, tanh) rather than in C++, so it is
visible and hackable at the same level as the rest of a user's forward()
-- one further application of "only the forward pass is ever written".
"""

from typing import List, NamedTuple, Optional, Tuple, Union

import cutagi
import numpy as np

ta = cutagi.tagi_autocov

# Re-exported as-is: plain ops already work directly on GaussianTensor.
GaussianTensor = ta.GaussianTensor
Parameter = ta.Parameter
tensor = ta.tensor
add = ta.add
mul = ta.mul
relu = ta.relu
tanh = ta.tanh
sigmoid = ta.sigmoid
mixture_relu = ta.mixture_relu
mixture_sigmoid = ta.mixture_sigmoid
mixture_tanh = ta.mixture_tanh
softplus = ta.softplus
leaky_relu = ta.leaky_relu
even_exp = ta.even_exp
chunk = ta.chunk
scale = ta.scale
gather = ta.gather
matmul = ta.matmul
softmax = ta.softmax
set_trace = ta.set_trace
clear_cov_tapes = ta.clear_cov_tapes
cross_cov = ta.cross_cov


def _pair(v) -> Tuple[int, int]:
    """Normalize an int-or-pair argument, like torch.nn.modules.utils._pair."""
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def _collect_parameters(value) -> List[Parameter]:
    if isinstance(value, Parameter):
        return [value]
    if hasattr(value, "parameters") and callable(value.parameters):
        return list(value.parameters())
    if isinstance(value, (list, tuple)):
        out: List[Parameter] = []
        for item in value:
            out.extend(_collect_parameters(item))
        return out
    return []


class Module:
    """Subclass and define forward(); parameters() recursively collects
    every Parameter reachable from self.__dict__ (nested Modules, lists/
    tuples of them, and the C++ Linear -- which already exposes its own
    .parameters())."""

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for value in self.__dict__.values():
            params.extend(_collect_parameters(value))
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """Thin wrapper around ta.Linear (weights initialized by
    param_init.cpp's init_weight_bias_linear, same as production). Also
    accepts a list/tuple of GaussianTensor (e.g. an LSTM's full output
    sequence) and applies itself to each element, mirroring how
    torch.nn.Linear broadcasts over any leading dimensions.

    `fan_in_override`, when > 0, is used in place of n_in only for the
    He/Xavier scale computation -- see LSTM below for why."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        bias: bool = True,
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        init_method: str = "He",
        fan_in_override: int = 0,
    ):
        self._impl = ta.Linear(
            n_in, n_out, bias, gain_w, gain_b, init_method, fan_in_override
        )

    @property
    def weight(self) -> Parameter:
        return self._impl.weight

    @property
    def bias(self) -> Optional[Parameter]:
        return self._impl.bias

    def parameters(self) -> List[Parameter]:
        return self._impl.parameters()

    def forward(
        self, x: Union[GaussianTensor, List[GaussianTensor]]
    ) -> Union[GaussianTensor, List[GaussianTensor]]:
        if isinstance(x, (list, tuple)):
            return [self._impl.forward(x_t) for x_t in x]
        return self._impl.forward(x)


class LSTMState(NamedTuple):
    h: GaussianTensor
    c: GaussianTensor


class LSTM(Module):
    """One LSTM layer (same per-layer semantics as torch.nn.LSTM(
    input_size, hidden_size, batch_first=True)) -- stack several LSTM
    instances for a multi-layer network, feeding one layer's output
    sequence as the next layer's input (see module docstring).

    forward(x, hx=None):
        x  : numpy array, shape (batch, seq_len, input_size), OR a list
             of GaussianTensor of length seq_len (e.g. another LSTM
             layer's output) -- so uncertainty propagates from layer to
             layer as GaussianTensors, never round-tripping through a
             plain numpy array between layers.
        hx : optional LSTMState to resume a truncated-BPTT rollout (e.g.
             returned by a previous forward() call); None starts from a
             fresh zero state.
        Returns (outputs, hx):
          outputs : list of GaussianTensor, length seq_len -- this
                    layer's full output sequence, torch-style. Index
                    outputs[-1] for a many-to-one head, or pass the
                    whole list into the next LSTM layer / a Linear head
                    (Linear broadcasts over the list, see above).
          hx      : final LSTMState for this layer, still attached to
                    the graph (h/c already have retain() set). To carry
                    filtered state into the next window: call observe()
                    on the final prediction first, THEN detach hx's h/c
                    -- observe() before detach.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        init_method: str = "He",
        track_cross_cov: bool = True,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Every gate reads the same [x_t, h_{t-1}], so i_t/g_t (and o_t vs
        # tanh(c_t)) are correlated. With track_cross_cov the engine's
        # covariance tape reconstructs those cross-covariances
        # automatically -- the autograd analogue of the hand-derived
        # lstm_cov_input_cell_states / lstm_cov_output_tanh_cell_states in
        # src/lstm_layer.cpp. False falls back to treating the gates as
        # independent (the engine's default), for A/B comparison.
        self.track_cross_cov = track_cross_cov

        # forget/input/candidate/output gates: z = Linear_x(x) + Linear_h(h_prev)
        # -- an affine map over the implicit concatenation [x, h_prev],
        # split into two independent Linears (only the x-side owns the
        # bias) so no separate "concat" op is needed. fan_in_override
        # makes each split Linear use the SAME scale a single fused
        # (input_size+hidden_size) -> hidden_size matrix would get under
        # param_init.cpp's init_weight_bias_lstm.
        fan_in = input_size + hidden_size
        gate_kwargs = dict(init_method=init_method, fan_in_override=fan_in)
        self.w_fx = Linear(input_size, hidden_size, **gate_kwargs)
        self.w_fh = Linear(hidden_size, hidden_size, bias=False, **gate_kwargs)
        self.w_ix = Linear(input_size, hidden_size, **gate_kwargs)
        self.w_ih = Linear(hidden_size, hidden_size, bias=False, **gate_kwargs)
        self.w_cx = Linear(input_size, hidden_size, **gate_kwargs)
        self.w_ch = Linear(hidden_size, hidden_size, bias=False, **gate_kwargs)
        self.w_ox = Linear(input_size, hidden_size, **gate_kwargs)
        self.w_oh = Linear(hidden_size, hidden_size, bias=False, **gate_kwargs)

    def zero_state(self, batch: int = 1, var: float = 0.0) -> LSTMState:
        """Deterministic zero initial state (mu=0, var=0), matching the
        production LSTM's reset_prev_states(). A nonzero var here injects
        fictitious uncertainty that inflates every gate variance in the
        window AND diverts part of each observation's innovation into
        updating the meaningless initial state instead of the weights --
        empirically this costs a large chunk of accuracy."""
        mu = [0.0] * (batch * self.hidden_size)
        v = [var] * (batch * self.hidden_size)
        return LSTMState(
            tensor(mu, [batch, self.hidden_size], v, "h0"),
            tensor(mu, [batch, self.hidden_size], v, "c0"),
        )

    def step(self, x_t: GaussianTensor, prev: LSTMState) -> LSTMState:
        """One time step:
        f_t = sigmoid(W_fx x_t + W_fh h_{t-1} + b_f)   forget gate
        i_t = sigmoid(W_ix x_t + W_ih h_{t-1} + b_i)   input gate
        g_t = tanh   (W_cx x_t + W_ch h_{t-1} + b_c)   candidate state
        o_t = sigmoid(W_ox x_t + W_oh h_{t-1} + b_o)   output gate
        c_t = f_t * c_{t-1} + i_t * g_t
        h_t = o_t * tanh(c_t)
        """
        if self.track_cross_cov:
            # This step's stochastic roots. Tapes are scoped to one step:
            # clear the previous step's (bounding memory), then re-mark --
            # track_cov() treats each root as an exogenous Gaussian with
            # its CURRENT marginal variance, exactly how the production
            # TAGI-LSTM treats h_{t-1}/c_{t-1}.
            clear_cov_tapes()
            x_t.track_cov()
            prev.h.track_cov()
            prev.c.track_cov()
        f = sigmoid(add(self.w_fx(x_t), self.w_fh(prev.h)))
        i = sigmoid(add(self.w_ix(x_t), self.w_ih(prev.h)))
        g = tanh(add(self.w_cx(x_t), self.w_ch(prev.h)))
        o = sigmoid(add(self.w_ox(x_t), self.w_oh(prev.h)))

        c = add(mul(f, prev.c), mul(i, g))
        h = mul(o, tanh(c))
        h.retain()
        c.retain()
        return LSTMState(h, c)

    def forward(
        self,
        x: Union[np.ndarray, List[GaussianTensor]],
        hx: Optional[LSTMState] = None,
    ) -> Tuple[List[GaussianTensor], LSTMState]:
        if isinstance(x, np.ndarray):
            batch, seq_len, _ = x.shape
            steps = [
                tensor(
                    x[:, t, :].astype(float).flatten().tolist(),
                    [batch, self.input_size],
                )
                for t in range(seq_len)
            ]
        else:
            steps = list(x)
            batch = steps[0].shape[0]

        if hx is None:
            hx = self.zero_state(batch)

        outputs = []
        for x_t in steps:
            hx = self.step(x_t, hx)
            outputs.append(hx.h)
        return outputs, hx

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for gate in (
            self.w_fx,
            self.w_fh,
            self.w_ix,
            self.w_ih,
            self.w_cx,
            self.w_ch,
            self.w_ox,
            self.w_oh,
        ):
            params.extend(gate.parameters())
        return params


class Sequential(Module):
    def __init__(self, *items):
        self.items = list(items)

    def parameters(self) -> List[Parameter]:
        params: List[Parameter] = []
        for it in self.items:
            params.extend(_collect_parameters(it))
        return params

    def forward(self, x):
        for it in self.items:
            x = it(x)
        return x


def _as_2d_tensor(x, cols_hint: Optional[int] = None) -> GaussianTensor:
    """Accept a numpy array (any leading dims, flattened to 2-D with the
    last axis as cols_hint) or an existing GaussianTensor, as torch layers
    accept both tensors and things convertible to them."""
    if isinstance(x, GaussianTensor):
        return x
    arr = np.asarray(x, dtype=float)
    cols = cols_hint if cols_hint is not None else arr.shape[-1]
    rows = arr.size // cols
    return tensor(arr.flatten().tolist(), [rows, cols])


class Conv2d(Module):
    """torch.nn.Conv2d equivalent: im2col (a gather, exact gain-1
    reindexing) followed by the existing linear() op, which is precisely
    how a convolution is a linear map with shared weights. Weight He init
    on fan_in = in_channels*kh*kw is identical to param_init.cpp's
    init_weight_bias_conv2d for the production Conv2d. The parameter
    update summing over all rows (batch x output positions) of the patch
    matrix reproduces weight sharing: every position's innovation
    accumulates into the same kernel, as in the production conv backward.

    forward(x, hw=None):
        x  : numpy (B, C, H, W) -- optionally uncertain via x_var= --
             or a GaussianTensor of shape {B, C*H*W} (channel-major,
             production cuTAGI image layout), in which case hw=(H, W)
             is required.
        Returns a GaussianTensor of shape {B, out_channels*oh*ow}; the
        spatial size is available as self.output_hw((H, W)).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        gain_w: float = 1.0,
        gain_b: float = 1.0,
        init_method: str = "He",
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        kh, kw = self.kernel_size
        self._lin = Linear(
            in_channels * kh * kw,
            out_channels,
            bias=bias,
            gain_w=gain_w,
            gain_b=gain_b,
            init_method=init_method,
        )
        self._idx_cache = {}

    @property
    def weight(self) -> Parameter:
        return self._lin.weight

    @property
    def bias(self) -> Optional[Parameter]:
        return self._lin.bias

    def output_hw(self, hw: Tuple[int, int]) -> Tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        return (hw[0] + 2 * ph - kh) // sh + 1, (hw[1] + 2 * pw - kw) // sw + 1

    def _indices(self, batch: int, hw: Tuple[int, int]):
        key = (batch, hw)
        if key in self._idx_cache:
            return self._idx_cache[key]
        H, W = hw
        C = self.in_channels
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh, ow = self.output_hw(hw)

        # Source row/col per (output position, kernel offset); -1 = padding
        si = np.arange(oh)[:, None] * sh - ph + np.arange(kh)[None, :]
        sj = np.arange(ow)[:, None] * sw - pw + np.arange(kw)[None, :]
        SI = si[:, None, :, None]  # (oh, 1, kh, 1)
        SJ = sj[None, :, None, :]  # (1, ow, 1, kw)
        valid = (SI >= 0) & (SI < H) & (SJ >= 0) & (SJ < W)
        spatial = SI * W + SJ  # (oh, ow, kh, kw)

        b = np.arange(batch)[:, None, None, None, None, None]
        c = np.arange(C)[None, None, None, :, None, None]
        idx = b * (C * H * W) + c * (H * W) + spatial[None, :, :, None, :, :]
        idx = np.where(valid[None, :, :, None, :, :], idx, -1)
        im2col = idx.reshape(batch * oh * ow, C * kh * kw)

        # Permute {B*oh*ow, out_c} -> {B, out_c*oh*ow} (torch NCHW layout)
        p = np.arange(oh * ow)
        co = np.arange(self.out_channels)
        perm = (
            np.arange(batch)[:, None, None] * (oh * ow) + p[None, None, :]
        ) * self.out_channels + co[None, :, None]
        perm = perm.reshape(batch, self.out_channels * oh * ow)

        cached = (
            im2col.flatten().tolist(),
            perm.flatten().tolist(),
            oh,
            ow,
        )
        self._idx_cache[key] = cached
        return cached

    def forward(
        self, x, hw: Optional[Tuple[int, int]] = None
    ) -> GaussianTensor:
        if not isinstance(x, GaussianTensor):
            arr = np.asarray(x, dtype=float)
            batch, C, H, W = arr.shape
            hw = (H, W)
            x = tensor(
                arr.reshape(batch, -1).flatten().tolist(), [batch, C * H * W]
            )
        else:
            if hw is None:
                raise ValueError(
                    "Conv2d: pass hw=(H, W) when x is a GaussianTensor"
                )
            batch = x.shape[0]
        kh, kw = self.kernel_size
        im2col_idx, perm_idx, oh, ow = self._indices(batch, hw)

        patches = gather(
            x, im2col_idx, batch * oh * ow, self.in_channels * kh * kw
        )
        z = self._lin(patches)  # {B*oh*ow, out_c}
        return gather(z, perm_idx, batch, self.out_channels * oh * ow)


class MaxPool2d(Module):
    """torch.nn.MaxPool2d equivalent. Selection by the maximum MEAN in
    each window at forward time, then a single gather routes that
    element's (mu, var) through with gain exactly 1 -- the same selection
    rule as the production MaxPool2d layer. Default stride equals
    kernel_size, matching torch.

    forward(x, hw=None) takes numpy (B, C, H, W) or a GaussianTensor
    {B, C*H*W} plus hw=(H, W); returns {B, C*oh*ow}."""

    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size
        self.padding = _pair(padding)

    def output_hw(self, hw: Tuple[int, int]) -> Tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        return (hw[0] + 2 * ph - kh) // sh + 1, (hw[1] + 2 * pw - kw) // sw + 1

    def forward(
        self,
        x,
        hw: Optional[Tuple[int, int]] = None,
        channels: Optional[int] = None,
    ) -> GaussianTensor:
        if not isinstance(x, GaussianTensor):
            arr = np.asarray(x, dtype=float)
            batch, C, H, W = arr.shape
            hw = (H, W)
            x = tensor(
                arr.reshape(batch, -1).flatten().tolist(), [batch, C * H * W]
            )
        else:
            if hw is None:
                raise ValueError(
                    "MaxPool2d: pass hw=(H, W) when x is a GaussianTensor"
                )
            batch = x.shape[0]
            C = (
                channels
                if channels is not None
                else x.shape[1] // (hw[0] * hw[1])
            )
        H, W = hw
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        oh, ow = self.output_hw(hw)

        si = np.arange(oh)[:, None] * sh - ph + np.arange(kh)[None, :]
        sj = np.arange(ow)[:, None] * sw - pw + np.arange(kw)[None, :]
        SI = si[:, None, :, None]
        SJ = sj[None, :, None, :]
        valid = (SI >= 0) & (SI < H) & (SJ >= 0) & (SJ < W)
        spatial = SI * W + SJ  # (oh, ow, kh, kw)

        b = np.arange(batch)[:, None, None, None, None, None]
        c = np.arange(C)[None, :, None, None, None, None]
        idx = b * (C * H * W) + c * (H * W) + spatial[None, None, :, :, :, :]
        idx = np.where(valid[None, None, :, :, :, :], idx, -1)
        idx = idx.reshape(batch, C, oh, ow, kh * kw)

        # Route the max-mu element of each window (production MaxPool2d rule)
        mu = np.asarray(x.mu)
        vals = np.where(idx >= 0, mu[np.maximum(idx, 0)], -np.inf)
        best = np.argmax(vals, axis=-1)
        chosen = np.take_along_axis(idx, best[..., None], axis=-1)[..., 0]
        chosen = chosen.reshape(batch, C * oh * ow)

        return gather(x, chosen.flatten().tolist(), batch, C * oh * ow)


def rms_norm_rows(x, eps: float = 1e-6):
    """Multiply each row by the deterministic constant 1/rms(row), where
    the RMS is over the Gaussian second moment E[x^2] = mu^2 + var.
    Treating the statistic as deterministic is the usual TAGI
    local-linearization convention (softmax max-subtraction, max-pooling
    index selection). Implemented with mul() against a var=0 tensor, so
    the backward gain is exact and the constant receives no update."""
    rows, cols = x.shape
    mu = np.array(x.mu).reshape(rows, cols)
    var = np.array(x.var).reshape(rows, cols)
    inv = 1.0 / np.sqrt(np.mean(mu**2 + var, axis=1, keepdims=True) + eps)
    c = np.broadcast_to(inv, (rows, cols)).flatten().tolist()
    return mul(x, tensor(c, [rows, cols], [], "rms_c"))


class MultiheadAttention(Module):
    """torch.nn.MultiheadAttention equivalent (batch_first=True), built
    from the engine's primitives:

        Q, K, V   = q_proj(x), k_proj(x), v_proj(x)      (linear op)
        scores    = (Q / sqrt(d_head)) K^T               (scale + matmul,
                     the GMA treatment of production query_key())
        attn      = softmax(scores)                      (production
                     softmax_mean_var, diagonal jcb; production MHA uses
                     Remax, whose cross-covariances this diagonal engine
                     cannot represent)
        out       = out_proj(attn V)                     (matmul + linear)

    Head split/merge are gathers (exact gain-1 reindexing). Uncertainty
    flows through every step and the backward sweep updates all four
    projections plus everything upstream -- no backward code anywhere.

    forward(query, key=None, value=None, batch_seq=None):
        query/key/value : numpy (B, S, E), or GaussianTensor {B*S, E}
             with batch_seq=(B, S) required. key/value default to query
             (self-attention).
        Returns (output {B*S, E}, attn_weights {B*H*S, S}), mirroring
        torch's (attn_output, attn_output_weights).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        init_method: str = "He",
        qk_norm: bool = False,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # QK-norm (Henry et al. 2020; ViT-22B): RMS-normalize each q/k row
        # before the dot product. Without it, TAGI training through
        # scores = Q K^T is a positive feedback loop -- every observe()
        # pushes BOTH factors in aligned directions, the logits grow, the
        # softmax saturates, and the run diverges (SGD only avoids this by
        # having a small learning rate). With qk_norm the logits are
        # bounded regardless of projection-weight magnitude. Off by
        # default: it changes the function, so exact parity with
        # torch.nn.MultiheadAttention holds only when disabled.
        self.qk_norm = qk_norm
        self.q_proj = Linear(
            embed_dim, embed_dim, bias=bias, init_method=init_method
        )
        self.k_proj = Linear(
            embed_dim, embed_dim, bias=bias, init_method=init_method
        )
        self.v_proj = Linear(
            embed_dim, embed_dim, bias=bias, init_method=init_method
        )
        self.out_proj = Linear(
            embed_dim, embed_dim, bias=bias, init_method=init_method
        )
        self._idx_cache = {}

    def _indices(self, batch: int, seq: int):
        key = (batch, seq)
        if key in self._idx_cache:
            return self._idx_cache[key]
        E, H, dh = self.embed_dim, self.num_heads, self.head_dim
        b = np.arange(batch)
        h = np.arange(H)
        s = np.arange(seq)
        d = np.arange(dh)

        # split[(b*H + h)*S + s, d] = x[(b*S + s)*E + h*dh + d]
        split = (
            (b[:, None, None, None] * seq + s[None, None, :, None]) * E
            + h[None, :, None, None] * dh
            + d[None, None, None, :]
        ).reshape(batch * H * seq, dh)

        # merge[(b*S + s)*E + h*dh + d] = ctx[((b*H + h)*S + s)*dh + d]
        merge = (
            (
                (b[:, None, None, None] * H + h[None, None, :, None]) * seq
                + s[None, :, None, None]
            )
            * dh
            + d[None, None, None, :]
        ).reshape(batch * seq, E)

        cached = (split.flatten().tolist(), merge.flatten().tolist())
        self._idx_cache[key] = cached
        return cached

    def forward(
        self,
        query,
        key=None,
        value=None,
        batch_seq: Optional[Tuple[int, int]] = None,
    ):
        if not isinstance(query, GaussianTensor):
            arr = np.asarray(query, dtype=float)
            batch, seq, _ = arr.shape
        else:
            if batch_seq is None:
                raise ValueError(
                    "MultiheadAttention: pass batch_seq=(B, S) when query "
                    "is a GaussianTensor"
                )
            batch, seq = batch_seq
        E, H, dh = self.embed_dim, self.num_heads, self.head_dim

        qt = _as_2d_tensor(query, E)
        kt = qt if key is None else _as_2d_tensor(key, E)
        vt = qt if value is None else _as_2d_tensor(value, E)

        split_idx, merge_idx = self._indices(batch, seq)
        rows_h = batch * H * seq

        q = gather(
            scale(self.q_proj(qt), 1.0 / float(np.sqrt(dh))),
            split_idx,
            rows_h,
            dh,
        )
        k = gather(self.k_proj(kt), split_idx, rows_h, dh)
        v = gather(self.v_proj(vt), split_idx, rows_h, dh)
        if self.qk_norm:
            q = rms_norm_rows(q)
            k = rms_norm_rows(k)

        scores = matmul(q, k, batch * H, seq, dh, seq, transpose_b=True)
        attn = softmax(scores)  # {B*H*S, S}
        ctx = matmul(attn, v, batch * H, seq, seq, dh)
        merged = gather(ctx, merge_idx, batch * seq, E)
        return self.out_proj(merged), attn
