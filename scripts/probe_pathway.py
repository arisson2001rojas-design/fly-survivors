"""Activate a set of cell types (Shiu et al. style Poisson drive) and watch what follows.

Useful to find where a pathway breaks in the LIF model, e.g.

    python scripts/probe_pathway.py --activate LC4,LPLC2 --side right
    python scripts/probe_pathway.py --activate T5a,T5b,T5c,T5d --side right
    python scripts/probe_pathway.py --activate L1,L2,L3 --side right --rate 150
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from flysurvivors import LIFBrain, load_connectome
from flysurvivors.annotations import Annotations
from flysurvivors.motor import DN_GROUPS, VISUAL_GROUPS, MotorReadout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activate", required=True, help="comma-separated cell types")
    ap.add_argument("--side", default=None, choices=[None, "left", "right"])
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--t-ms", type=float, default=500.0)
    ap.add_argument("--fraction", type=float, default=1.0, help="drive a random subset")
    ap.add_argument("--top", type=int, default=15, help="also list the most active cell types")
    args = ap.parse_args()

    cn = load_connectome()
    ann = Annotations(cn)
    types = args.activate.split(",")
    idx = ann.select(types, args.side)
    if args.fraction < 1:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, int(len(idx) * args.fraction), replace=False)
    print(f"activating {len(idx)} neurons of {types} ({args.side or 'both'}) at {args.rate} Hz")

    brain = LIFBrain(cn)
    brain.set_stimulus(idx, args.rate)
    groups = {**ann.groups(VISUAL_GROUPS), **ann.groups(DN_GROUPS)}
    readout = MotorReadout(groups, brain.device, tau_ms=1e9)  # plain average over the run
    counts_all, _ = brain.run(args.t_ms)
    torch.cuda.synchronize()
    rates = counts_all / (args.t_ms / 1000.0)
    print("group mean rates (Hz/neuron):")
    for name, g in groups.items():
        r = rates[g].mean()
        if r > 0:
            print(f"  {name:10s} {r:7.1f}   ({(rates[g] > 0).mean() * 100:.0f}% active)")

    stim = np.zeros(cn.n_neurons, bool)
    stim[idx] = True
    df = ann.df.copy()
    df["rate"] = rates[df.index]
    df = df[(df["rate"] > 0) & ~stim[df.index]]
    print(f"active non-stimulated neurons: {len(df)}")
    top = (
        df.groupby(["cell_type", "side"])["rate"]
        .agg(["mean", "size"])
        .sort_values("mean", ascending=False)
        .head(args.top)
    )
    print(top.round(1).to_string())


if __name__ == "__main__":
    main()
