"""Validation experiment from Shiu et al. 2024: sugar GRNs -> proboscis extension (MN9).

Activates the right-hemisphere sugar-sensing neurons with Poisson spiking and checks that
the proboscis motor neuron MN9 fires. Reports MN9 rate, number of active neurons and the
most active neurons, for comparison with the reference notebook.
"""

import argparse
import time

import numpy as np
import torch

from flysurvivors import LIFBrain, load_connectome
from flysurvivors.neurons import MN9, SUGAR_GRN_RIGHT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=200.0, help="stimulation rate (Hz)")
    ap.add_argument("--t-ms", type=float, default=1000.0, help="trial duration (ms)")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--min-syn", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-graph", action="store_true")
    args = ap.parse_args()

    cn = load_connectome(min_syn=args.min_syn)
    print(cn.summary())
    sugar_idx, missing = cn.index_of_existing(SUGAR_GRN_RIGHT)
    print(f"sugar GRNs: {len(sugar_idx)} found, {len(missing)} missing from v783")
    mn9_idx = int(cn.index_of([MN9])[0])

    brain = LIFBrain(cn, device=args.device)
    brain.set_stimulus(sugar_idx, args.rate)

    rates = []
    mn9_rates = []
    for k in range(args.trials):
        brain.reset()
        t0 = time.perf_counter()
        counts, _ = brain.run(args.t_ms, use_graph=not args.no_graph)
        if args.device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        r = counts / (args.t_ms / 1000.0)
        rates.append(r)
        mn9_rates.append(r[mn9_idx])
        n_active = int((counts > 0).sum())
        print(
            f"trial {k + 1}: {wall:.2f} s wall for {args.t_ms:.0f} ms bio "
            f"({args.t_ms / 1000 / wall:.2f}x realtime), {n_active} active neurons, "
            f"MN9 = {r[mn9_idx]:.1f} Hz"
        )

    mean_rate = np.mean(rates, axis=0)
    stim = np.zeros(cn.n_neurons, dtype=bool)
    stim[sugar_idx] = True
    print()
    print(f"MN9 ({MN9}): {np.mean(mn9_rates):.1f} +/- {np.std(mn9_rates):.1f} Hz")
    print(f"active non-stimulated neurons (mean rate > 0): {int(((mean_rate > 0) & ~stim).sum())}")
    print("top 15 non-stimulated neurons:")
    order = np.argsort(-np.where(stim, -1.0, mean_rate))[:15]
    for i in order:
        tag = " <- MN9" if i == mn9_idx else ""
        print(f"  {cn.ids[i]}  {mean_rate[i]:7.1f} Hz{tag}")


if __name__ == "__main__":
    main()
