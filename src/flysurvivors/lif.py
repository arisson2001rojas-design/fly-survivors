"""Leaky integrate-and-fire whole-brain model on the GPU (PyTorch + Triton).

Reimplements the Brian2 model of Shiu et al. 2024 with a fixed time step:

    dv/dt = (v_rest - v + g) / tau_m      (frozen while refractory)
    dg/dt = -g / tau_syn                  (frozen while refractory)
    spike when v > v_th  ->  v = v_reset, g = 0, refractory for t_refr
    presynaptic spike    ->  g_post += w_syn * signed_synapse_count, after a delay
                             (dropped if the postsynaptic neuron is refractory)

"Optogenetic" activation of a neuron set is modelled as Poisson spiking at a fixed rate
(the neuron is forced above threshold, with no refractory period), like the reference.

Two backends with identical semantics:

- ``torch``: plain tensor ops and a CSR sparse mat-vec. Works on CPU; the reference.
- ``triton``: event-driven propagation + fused update kernels (see kernels.py). Faster.

Neither has host-side control flow inside :meth:`LIFBrain.step`, so a whole step can be
captured in a CUDA graph and replayed for real-time use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .connectome import Connectome
from .kernels import HAS_TRITON


@dataclass
class LIFParams:
    v_rest: float = -52.0  # mV
    v_reset: float = -52.0  # mV
    v_th: float = -45.0  # mV
    tau_m: float = 20.0  # ms, membrane time constant
    tau_syn: float = 5.0  # ms, synaptic time constant
    t_refr: float = 2.2  # ms, refractory period
    delay: float = 1.8  # ms, synaptic delay
    w_syn: float = 0.275  # mV per synapse
    dt: float = 0.1  # ms


class LIFBrain:
    """Whole-brain LIF simulator.

    Parameters
    ----------
    cn:
        Connectome to simulate.
    params:
        Model constants.
    device:
        ``"cuda"`` or ``"cpu"``.
    backend:
        ``"auto"`` (triton when available on CUDA, else torch), ``"triton"`` or ``"torch"``.
    """

    def __init__(
        self,
        cn: Connectome,
        params: LIFParams | None = None,
        device: str | torch.device = "cuda",
        backend: str = "auto",
    ) -> None:
        self.p = params or LIFParams()
        self.device = torch.device(device)
        self.n = cn.n_neurons
        p = self.p

        triton_ok = HAS_TRITON and self.device.type == "cuda"
        if backend == "auto":
            backend = "triton" if triton_ok else "torch"
        if backend == "triton" and not triton_ok:
            raise RuntimeError("triton backend needs Triton and a CUDA device")
        if backend not in ("triton", "torch"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend

        weight_mv = cn.weight * np.float32(p.w_syn)
        if backend == "triton":
            # CSR by presynaptic neuron for event-driven propagation.
            order = np.argsort(cn.pre, kind="stable")
            pre_s = cn.pre[order].astype(np.int64)
            rowptr = np.zeros(self.n + 1, dtype=np.int32)
            np.cumsum(np.bincount(pre_s, minlength=self.n), out=rowptr[1:])
            self.rowptr = torch.from_numpy(rowptr).to(self.device)
            self.col = torch.from_numpy(cn.post[order].astype(np.int32)).to(self.device)
            self.val = torch.from_numpy(weight_mv[order].astype(np.float32)).to(self.device)
        else:
            # W[post, pre] so that g_inc = W @ spikes.
            idx = torch.from_numpy(np.stack([cn.post.astype(np.int64), cn.pre.astype(np.int64)]))
            val = torch.from_numpy(weight_mv.astype(np.float32))
            w = torch.sparse_coo_tensor(idx, val, (self.n, self.n)).coalesce()
            self.W = w.to_sparse_csr().to(self.device)

        self.decay_m = math.exp(-p.dt / p.tau_m)
        self.decay_s = math.exp(-p.dt / p.tau_syn)
        self.refr_steps = int(round(p.t_refr / p.dt))
        self.delay_steps = int(round(p.delay / p.dt))
        self.hist_len = self.delay_steps + 1

        f32 = dict(dtype=torch.float32, device=self.device)
        i32 = dict(dtype=torch.int32, device=self.device)
        self.v = torch.full((self.n,), p.v_rest, **f32)
        self.g = torch.zeros(self.n, **f32)
        self.g_inc = torch.zeros(self.n, **f32)
        self.refr = torch.zeros(self.n, **i32)
        # Per-neuron refractory length (0 for Poisson-driven neurons, like the reference).
        self.refr_len = torch.full((self.n,), self.refr_steps, **i32)
        # Probability of a forced spike per step for Poisson-driven neurons.
        self.stim_prob = torch.zeros(self.n, **f32)
        # Ring buffer of past spike rows for the synaptic delay; hist_pos = row of this step.
        self.hist = torch.zeros(self.hist_len, self.n, **f32)
        self.hist_pos = torch.zeros((), dtype=torch.int64, device=self.device)
        self.spikes = torch.zeros(self.n, **f32)  # 1.0 where the neuron spiked this step
        self.seed = torch.zeros((), **i32)
        self.t_step = 0
        self._graph: torch.cuda.CUDAGraph | None = None

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.v.fill_(self.p.v_rest)
        self.g.zero_()
        self.g_inc.zero_()
        self.refr.zero_()
        self.hist.zero_()
        self.hist_pos.zero_()
        self.spikes.zero_()
        self.t_step = 0

    def set_stimulus(self, idx, rate_hz: float) -> None:
        """Drive neurons ``idx`` (model indices) with Poisson spikes at ``rate_hz``."""
        idx = torch.as_tensor(np.array(idx, dtype=np.int64), device=self.device)
        self.stim_prob[idx] = rate_hz * self.p.dt / 1000.0
        self.refr_len[idx] = 0

    def clear_stimulus(self) -> None:
        self.stim_prob.zero_()
        self.refr_len.fill_(self.refr_steps)

    # ------------------------------------------------------------------- step
    def _step_torch(self) -> None:
        p = self.p
        self.hist_pos.copy_(torch.remainder(self.hist_pos + 1, self.hist_len))
        active = self.refr == 0

        # Delayed presynaptic spikes arrive now.
        read_row = torch.remainder(self.hist_pos - self.delay_steps, self.hist_len)
        s_del = self.hist.index_select(0, read_row.view(1)).view(-1)
        g_inc = torch.mv(self.W, s_del)

        # Integrate (exact exponential decay of g; g held constant within the step for v).
        g_new = self.g * self.decay_s
        v_new = p.v_rest + g_new + (self.v - p.v_rest - g_new) * self.decay_m
        # Refractory neurons are frozen and drop incoming input, as in the Brian2 reference.
        g_int = torch.where(active, g_new + g_inc, self.g)
        v_int = torch.where(active, v_new, self.v)

        # Poisson drive: force the neuron above threshold.
        forced = torch.rand(self.n, device=self.device) < self.stim_prob
        v_int = torch.where(forced, torch.full_like(v_int, p.v_th + 1.0), v_int)

        spikes = (v_int > p.v_th) & active
        self.v.copy_(torch.where(spikes, torch.full_like(v_int, p.v_reset), v_int))
        self.g.copy_(torch.where(spikes, torch.zeros_like(g_int), g_int))
        self.refr.copy_(torch.where(spikes, self.refr_len, torch.clamp(self.refr - 1, min=0)))

        self.spikes.copy_(spikes)
        self.hist.index_copy_(0, self.hist_pos.view(1), self.spikes.view(1, -1))

    def _step_triton(self) -> None:
        from .kernels import lif_update_kernel, propagate_grouped_kernel

        p = self.p
        self.hist_pos.copy_(torch.remainder(self.hist_pos + 1, self.hist_len))
        self.g_inc.zero_()
        propagate_grouped_kernel[((self.n + 7) // 8,)](
            self.hist,
            self.hist_pos,
            self.rowptr,
            self.col,
            self.val,
            self.g_inc,
            self.n,
            self.delay_steps,
            self.hist_len,
            BLOCK=128,
            GROUP=8,
        )
        self.seed.add_(1)
        block = 1024
        lif_update_kernel[((self.n + block - 1) // block,)](
            self.v,
            self.g,
            self.refr,
            self.g_inc,
            self.stim_prob,
            self.refr_len,
            self.spikes,
            self.hist,
            self.hist_pos,
            self.seed,
            self.n,
            self.decay_m,
            self.decay_s,
            p.v_rest,
            p.v_th,
            p.v_reset,
            BLOCK=block,
        )

    def _step_impl(self) -> None:
        if self.backend == "triton":
            self._step_triton()
        else:
            self._step_torch()

    def step(self) -> torch.Tensor:
        """Advance one time step. Returns the (N,) float32 spike vector (device tensor)."""
        if self._graph is not None:
            self._graph.replay()
        else:
            self._step_impl()
        self.t_step += 1
        return self.spikes

    def capture_graph(self) -> None:
        """Capture :meth:`step` into a CUDA graph (removes Python launch overhead)."""
        if self.device.type != "cuda":
            return
        # Warm-up on a side stream as recommended by the PyTorch docs (also JIT-compiles Triton).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_impl()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_impl()
        self._graph = g
        self.reset()

    # -------------------------------------------------------------------- run
    def run_steps(self, n_steps: int, idx: torch.Tensor | None = None) -> torch.Tensor:
        """Advance ``n_steps`` and return spike counts (device tensor) for ``idx``
        (all neurons if None). Meant for the real-time loop: no host sync."""
        if self.device.type == "cuda" and self._graph is None:
            self.capture_graph()
        n = self.n if idx is None else len(idx)
        counts = torch.zeros(n, dtype=torch.float32, device=self.device)
        for _ in range(n_steps):
            s = self.step()
            counts += s if idx is None else s.index_select(0, idx)
        return counts

    def run(
        self,
        t_ms: float,
        record_idx=None,
        use_graph: bool = True,
    ) -> tuple[np.ndarray, list[tuple[int, float]]]:
        """Simulate ``t_ms`` milliseconds from the current state.

        Returns
        -------
        counts:
            (N,) spike counts per neuron over the run.
        events:
            ``(model_index, time_ms)`` for neurons in ``record_idx`` (empty if None).
        """
        if use_graph and self.device.type == "cuda" and self._graph is None:
            self.capture_graph()
        n_steps = int(round(t_ms / self.p.dt))
        counts = torch.zeros(self.n, dtype=torch.float32, device=self.device)
        rec_idx = None
        rec_buf = []
        if record_idx is not None:
            rec_idx = torch.as_tensor(np.asarray(record_idx, dtype=np.int64), device=self.device)
        t0 = self.t_step
        for _ in range(n_steps):
            s = self.step()
            counts += s
            if rec_idx is not None:
                rec_buf.append(s[rec_idx].clone())
        events: list[tuple[int, float]] = []
        if rec_idx is not None and rec_buf:
            m = torch.stack(rec_buf).cpu().numpy() > 0  # (steps, len(rec_idx))
            t_idx, n_idx = np.nonzero(m)
            rec_np = rec_idx.cpu().numpy()
            events = [(int(rec_np[j]), float((t0 + i) * self.p.dt)) for i, j in zip(t_idx, n_idx)]
        return counts.cpu().numpy(), events
