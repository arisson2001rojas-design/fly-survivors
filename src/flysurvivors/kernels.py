"""Triton kernels for the GPU fast path.

Two kernels per step, both free of host-side control flow so the whole step can be
captured in a CUDA graph:

- ``propagate``: event-driven synaptic propagation. One program per presynaptic neuron;
  it exits immediately unless that neuron spiked ``delay`` steps ago, otherwise it
  scatters its out-edges (CSR by presynaptic index) into ``g_inc`` with atomic adds.
  Cost scales with the number of spikes, not with the 15 M edges.
- ``lif_update``: the fused element-wise LIF update (decay, integrate, Poisson forcing,
  threshold, reset, refractory countdown) that writes the new spike row into the delay
  ring buffer.

The reference semantics are the plain-torch path in :mod:`flysurvivors.lif`; a test
checks the two agree exactly on a deterministic drive.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the platform
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def propagate_kernel(
        hist_ptr,  # (L, N) float32 ring buffer of spike rows
        pos_ptr,  # () int64 current write position in the ring buffer
        rowptr_ptr,  # (N+1,) int32 CSR row pointers, rows = presynaptic neuron
        col_ptr,  # (E,) int32 postsynaptic index
        val_ptr,  # (E,) float32 weight in mV
        out_ptr,  # (N,) float32 accumulator (zeroed by the caller)
        n,
        delay_steps,
        hist_len,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pos = tl.load(pos_ptr)
        read_row = (pos - delay_steps + hist_len) % hist_len
        s = tl.load(hist_ptr + read_row * n + pid)
        if s != 0.0:
            start = tl.load(rowptr_ptr + pid)
            end = tl.load(rowptr_ptr + pid + 1)
            for off in range(start, end, BLOCK):
                idx = off + tl.arange(0, BLOCK)
                mask = idx < end
                cols = tl.load(col_ptr + idx, mask=mask, other=0)
                vals = tl.load(val_ptr + idx, mask=mask, other=0.0)
                tl.atomic_add(out_ptr + cols, vals, mask=mask)

    @triton.jit
    def propagate_grouped_kernel(
        hist_ptr, pos_ptr, rowptr_ptr, col_ptr, val_ptr, out_ptr,
        n: tl.constexpr, delay_steps: tl.constexpr, hist_len: tl.constexpr,
        BLOCK: tl.constexpr, GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base = pid * GROUP
        pos = tl.load(pos_ptr)
        read_row = (pos - delay_steps + hist_len) % hist_len
        for j in tl.static_range(0, GROUP):
            nid = base + j
            if nid < n:
                s = tl.load(hist_ptr + read_row * n + nid)
                if s != 0.0:
                    start = tl.load(rowptr_ptr + nid)
                    end = tl.load(rowptr_ptr + nid + 1)
                    for off in range(start, end, BLOCK):
                        idx = off + tl.arange(0, BLOCK)
                        mask = idx < end
                        cols = tl.load(col_ptr + idx, mask=mask, other=0)
                        vals = tl.load(val_ptr + idx, mask=mask, other=0.0)
                        tl.atomic_add(out_ptr + cols, vals, mask=mask)

    @triton.jit
    def lif_update_kernel(
        v_ptr,
        g_ptr,
        refr_ptr,
        g_inc_ptr,
        stim_ptr,
        refr_len_ptr,
        spikes_ptr,  # (N,) float32 output: 1.0 where the neuron spiked
        hist_ptr,
        pos_ptr,
        seed_ptr,  # () int32, incremented by the caller every step
        n,
        decay_m,
        decay_s,
        v_rest,
        v_th,
        v_reset,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        v = tl.load(v_ptr + idx, mask=mask, other=0.0)
        g = tl.load(g_ptr + idx, mask=mask, other=0.0)
        refr = tl.load(refr_ptr + idx, mask=mask, other=1)
        g_inc = tl.load(g_inc_ptr + idx, mask=mask, other=0.0)
        stim = tl.load(stim_ptr + idx, mask=mask, other=0.0)
        refr_len = tl.load(refr_len_ptr + idx, mask=mask, other=0)

        active = refr == 0
        g_new = g * decay_s
        v_new = v_rest + g_new + (v - v_rest - g_new) * decay_m
        g_i = tl.where(active, g_new + g_inc, g)  # refractory: frozen, input dropped
        v_i = tl.where(active, v_new, v)

        seed = tl.load(seed_ptr)
        forced = tl.rand(seed, idx) < stim
        v_i = tl.where(forced, v_th + 1.0, v_i)

        spk = (v_i > v_th) & active
        tl.store(v_ptr + idx, tl.where(spk, v_reset, v_i), mask=mask)
        tl.store(g_ptr + idx, tl.where(spk, 0.0, g_i), mask=mask)
        tl.store(refr_ptr + idx, tl.where(spk, refr_len, tl.maximum(refr - 1, 0)), mask=mask)

        spk_f = spk.to(tl.float32)
        tl.store(spikes_ptr + idx, spk_f, mask=mask)
        pos = tl.load(pos_ptr)
        tl.store(hist_ptr + pos * n + idx, spk_f, mask=mask)
