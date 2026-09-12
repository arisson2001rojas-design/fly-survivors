"""Measure simulation speed: wall time per step and real-time ratio."""

import argparse
import time

import torch

from flysurvivors import LIFBrain, load_connectome
from flysurvivors.neurons import SUGAR_GRN_RIGHT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-syn", type=int, default=1)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cn = load_connectome(min_syn=args.min_syn)
    print(cn.summary())
    brain = LIFBrain(cn, device=args.device)
    idx, _ = cn.index_of_existing(SUGAR_GRN_RIGHT)
    brain.set_stimulus(idx, 200.0)

    for label, graph in (("eager", False), ("cuda graph", True)):
        if graph and args.device != "cuda":
            continue
        brain.reset()
        brain._graph = None
        if graph:
            brain.capture_graph()
        for _ in range(200):
            brain.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.steps):
            brain.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        bio_ms = args.steps * brain.p.dt
        print(
            f"{label:10s}: {wall / args.steps * 1e6:7.1f} us/step, "
            f"{bio_ms / 1000 / wall:.2f}x realtime"
        )


if __name__ == "__main__":
    main()
