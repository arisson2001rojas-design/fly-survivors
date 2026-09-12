"""Unit tests on a tiny synthetic connectome (CPU)."""

import numpy as np
import pytest

from flysurvivors import Connectome, LIFBrain, LIFParams


def make_cn(edges):
    """edges: list of (pre, post, signed_syn_count)."""
    n = 1 + max(max(a, b) for a, b, _ in edges)
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    pre = np.array([e[0] for e in edges], dtype=np.int32)
    post = np.array([e[1] for e in edges], dtype=np.int32)
    w = np.array([e[2] for e in edges], dtype=np.float32)
    return Connectome(ids, pre, post, w)


def run_counts(brain, t_ms):
    counts, _ = brain.run(t_ms, use_graph=False)
    return counts


def test_index_of():
    cn = make_cn([(0, 1, 5)])
    assert cn.index_of([1001, 1000]).tolist() == [1, 0]
    with pytest.raises(KeyError):
        cn.index_of([42])
    found, missing = cn.index_of_existing([1001, 42])
    assert found.tolist() == [1] and missing == [42]


def test_silent_without_input():
    cn = make_cn([(0, 1, 50)])
    brain = LIFBrain(cn, device="cpu")
    assert run_counts(brain, 100).sum() == 0


def test_poisson_drive_rate():
    cn = make_cn([(0, 1, 1)])
    brain = LIFBrain(cn, device="cpu")
    brain.set_stimulus([0], 200.0)
    counts = run_counts(brain, 5000)
    assert 800 < counts[0] < 1200  # 200 Hz over 5 s


def test_strong_excitation_propagates_and_inhibition_does_not():
    # 0 -> 1 excitatory, 200 synapses = 55 mV per spike; filtered by tau_syn / tau_m the
    # peak depolarisation of one spike is ~8.7 mV, just above the 7 mV threshold distance.
    # 0 -> 2 inhibitory with the same strength.
    cn = make_cn([(0, 1, 200), (0, 2, -200)])
    brain = LIFBrain(cn, device="cpu")
    brain.set_stimulus([0], 50.0)
    counts = run_counts(brain, 2000)
    assert 60 < counts[1] < 140  # roughly one output spike per input spike
    assert counts[2] == 0


def test_weak_excitation_needs_summation():
    # 1 synapse = 0.275 mV: a single 20 Hz input never reaches threshold,
    # but 20 concurrent presynaptic neurons at 100 Hz do.
    cn = make_cn([(0, 1, 1)])
    brain = LIFBrain(cn, device="cpu")
    brain.set_stimulus([0], 20.0)
    assert run_counts(brain, 2000)[1] == 0

    cn = make_cn([(i, 20, 4) for i in range(20)])
    brain = LIFBrain(cn, device="cpu")
    brain.set_stimulus(list(range(20)), 100.0)
    assert run_counts(brain, 2000)[20] > 10


def test_synaptic_delay():
    # 6000 synapses = 1650 mV: one integration step after delivery crosses threshold,
    # so neuron 1 fires exactly one step after the delayed spike arrives.
    cn = make_cn([(0, 1, 6000)])
    p = LIFParams()
    brain = LIFBrain(cn, p, device="cpu")
    brain.set_stimulus([0], 1e9)  # spike every step
    _, events = brain.run(5, record_idx=[0, 1], use_graph=False)
    t_first = {i: t for i, t in reversed(events)}
    assert t_first[0] == pytest.approx(0.0)
    assert t_first[1] == pytest.approx(p.delay + p.dt, abs=p.dt * 0.01)


def test_refractory_caps_rate():
    cn = make_cn([(0, 1, 500)])  # every input spike alone crosses threshold
    brain = LIFBrain(cn, device="cpu")
    brain.set_stimulus([0], 1000.0)
    counts = run_counts(brain, 1000)
    # Refractory 2.2 ms + 1 step caps the rate at ~435 Hz; inputs arriving while
    # refractory are dropped, so the neuron then waits ~1 ms for the next input.
    assert 150 < counts[1] <= 1000 / 2.3 + 1


def test_input_during_refractory_is_dropped():
    # Two strong inputs 1 ms apart: the second lands during the refractory period
    # opened by the first and must not produce a second spike (Brian2 semantics).
    cn = make_cn([(0, 1, 6000)])
    brain = LIFBrain(cn, device="cpu")
    brain.refr_len[0] = 0
    spikes = []
    for k in range(80):
        brain.stim_prob[0] = 1.0 if k in (0, 10) else 0.0
        if brain.step()[1] > 0:
            spikes.append(k * 0.1)
    assert spikes == [pytest.approx(1.9)]


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="needs CUDA"
)
def test_triton_matches_torch_on_full_brain():
    """Both backends must agree on a deterministic drive (Poisson prob = 1 every step)."""
    import torch

    from flysurvivors import load_connectome
    from flysurvivors.kernels import HAS_TRITON
    from flysurvivors.neurons import SUGAR_GRN_RIGHT

    if not HAS_TRITON:
        pytest.skip("needs Triton")
    cn = load_connectome()
    idx, _ = cn.index_of_existing(SUGAR_GRN_RIGHT)
    counts = {}
    for backend in ("torch", "triton"):
        brain = LIFBrain(cn, device="cuda", backend=backend)
        brain.set_stimulus(idx, 10_000.0)  # prob = 1.0 per step -> deterministic
        counts[backend], _ = brain.run(200.0)
        torch.cuda.synchronize()
    a, b = counts["torch"], counts["triton"]
    assert a.sum() > 1000
    # Float atomics reorder sums, and the network is chaotic, so a few threshold flips
    # cascade: measured ~1% of spikes differ under this extreme drive. Catch real bugs only.
    assert np.abs(a - b).sum() <= 0.03 * a.sum()
