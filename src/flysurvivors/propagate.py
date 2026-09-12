"""Spike propagation: g_post += w for every out-edge of every neuron that spiked.

Two backends:

- ``spmv``: ``torch.mv`` on a CSR matrix W[post, pre]. Simple, but touches all 15 M edges
  every step even though only a few dozen neurons spike per 0.1 ms.
- ``triton``: event-driven kernel. One program per presynaptic neuron; programs whose
  neuron did not spike exit immediately, the others scatter their out-edges with atomic
  adds. Cost scales with the number of spikes, not with the size of the connectome.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the platform
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _propagate_kernel(
        spikes_ptr,  # (N,) float32, 1.0 where the neuron spiked
        rowptr_ptr,  # (N+1,) int32 CSR row pointers, rows = presynaptic neuron
        col_ptr,  # (E,) int32 postsynaptic index
        val_ptr,  # (E,) float32 weight in mV
        out_ptr,  # (N,) float32 accumulator
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        s = tl.load(spikes_ptr + pid)
        if s != 0.0:
            start = tl.load(rowptr_ptr + pid)
            end = tl.load(rowptr_ptr + pid + 1)
            for off in range(start, end, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                mask = idx < end
                cols = tl.load(col_ptr + idx, mask=mask, other=0)
                vals = tl.load(val_ptr + idx, mask=mask, other=0.0)
                tl.atomic_add(out_ptr + cols, vals, mask=mask)


class Propagator:
    """Turns a spike vector into the synaptic increment vector."""

    def __init__(self, pre, post, weight_mv, n: int, device: torch.device, backend: str = "auto"):
        self.n = n
        self.device = device
        if backend == "auto":
            backend = "triton" if (HAS_TRITON and device.type == "cuda") else "spmv"
        if backend == "triton" and not (HAS_TRITON and device.type == "cuda"):
            raise RuntimeError("triton backend needs Triton and a CUDA device")
        self.backend = backend

        if backend == "triton":
            order = np.argsort(pre, kind="stable")
            pre_s = pre[order].astype(np.int64)
            rowptr = np.zeros(n + 1, dtype=np.int32)
            np.cumsum(np.bincount(pre_s, minlength=n), out=rowptr[1:])
            self.rowptr = torch.from_numpy(rowptr).to(device)
            self.col = torch.from_numpy(post[order].astype(np.int32)).to(device)
            self.val = torch.from_numpy(weight_mv[order].astype(np.float32)).to(device)
            self.out = torch.zeros(n, dtype=torch.float32, device=device)
            self.block = 128
        else:
            idx = torch.from_numpy(np.stack([post.astype(np.int64), pre.astype(np.int64)]))
            val = torch.from_numpy(weight_mv.astype(np.float32))
            w = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
            self.W = w.to_sparse_csr().to(device)

    def __call__(self, spikes_f32: torch.Tensor) -> torch.Tensor:
        """``spikes_f32``: (N,) float32 0/1. Returns (N,) float32 increments in mV."""
        if self.backend == "triton":
            self.out.zero_()
            _propagate_kernel[(self.n,)](
                spikes_f32, self.rowptr, self.col, self.val, self.out, BLOCK=self.block
            )
            return self.out
        return torch.mv(self.W, spikes_f32)
