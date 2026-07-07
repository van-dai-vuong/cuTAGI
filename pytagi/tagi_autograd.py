"""Torch-like Python layer on top of ``cutagi.tagi_autograd`` (the pybind11
binding of include/tagi_autograd.h). You define only the forward pass by
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

The LSTM gate math is written here in Python from the tagi_autograd
primitives (Linear, add, mul, sigmoid, tanh) rather than in C++, so it is
visible and hackable at the same level as the rest of a user's forward()
-- one further application of "only the forward pass is ever written".
"""

from typing import List, NamedTuple, Optional, Tuple, Union

import cutagi
import numpy as np

ta = cutagi.tagi_autograd

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
set_trace = ta.set_trace


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
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size

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
