"""End-to-end check of eye + brain + motor readout on a synthetic scene.

A dark disc (an "enemy") approaches the fly on a bright ground from a chosen azimuth.
We watch the looming detectors (LC4, LPLC2), the giant fiber and the walking
descending neurons, and what the locomotion layer would send to the gamepad.

    python scripts/demo_looming.py --azimuth 90      # from the right
    python scripts/demo_looming.py --azimuth 0       # head on
    python scripts/demo_looming.py --no-enemy        # baseline, ground only
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from flysurvivors import LIFBrain, load_connectome
from flysurvivors.annotations import Annotations
from flysurvivors.columns import load_eye_columns
from flysurvivors.eye import DriveParams, EyeParams, RetinaDrive, VirtualEye
from flysurvivors.motor import DN_GROUPS, VISUAL_GROUPS, Locomotion, MotorReadout


def render(size, px_per_unit, enemy_xy, radius, ground=0.7, enemy=0.05) -> np.ndarray:
    """Player-centred grayscale crop, heading = +x. ``enemy_xy`` in world units or None."""
    img = np.full((size, size), ground, dtype=np.float32)
    if enemy_xy is not None:
        c = (size - 1) / 2
        yy, xx = np.mgrid[0:size, 0:size]
        ex, ey = c + enemy_xy[0] * px_per_unit, c + enemy_xy[1] * px_per_unit
        img[(xx - ex) ** 2 + (yy - ey) ** 2 <= (radius * px_per_unit) ** 2] = enemy
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--azimuth", type=float, default=90.0, help="approach direction (deg)")
    ap.add_argument("--no-enemy", action="store_true")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--start", type=float, default=7.0, help="start distance (world units)")
    ap.add_argument("--speed", type=float, default=3.0, help="approach speed (units/s)")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--rate", type=float, default=150.0, help="max input rate (Hz)")
    ap.add_argument("--tonic", type=float, default=1.0)
    ap.add_argument("--phasic", type=float, default=0.0)
    ap.add_argument("--baseline", type=float, default=0.0)
    ap.add_argument("--adapt-ms", type=float, default=300.0)
    ap.add_argument("--drive", default=None,
                    help="cell types to drive with polarity, e.g. 'L1:-1,L2:-1,Tm1:-1' "
                         "(default: flysurvivors.eye.DEFAULT_DRIVE_TYPES)")
    ap.add_argument("--absolute", action="store_true", help="absolute instead of relative contrast")
    ap.add_argument("--every", type=float, default=0.1, help="print period (s)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--plot", default=None, help="save a PNG of the rates")
    args = ap.parse_args()

    cn = load_connectome()
    ann = Annotations(cn)
    columns = load_eye_columns(cn)
    brain = LIFBrain(cn)
    eye = VirtualEye(columns, EyeParams())
    drive_types = None
    if args.drive:
        drive_types = {k: float(v) for k, v in (kv.split(":") for kv in args.drive.split(","))}
    drive = RetinaDrive(
        brain, columns, cell_types=drive_types, active_columns=eye.sees_ground,
        params=DriveParams(rate_max_hz=args.rate, tonic_gain=args.tonic, phasic_gain=args.phasic,
                           baseline=args.baseline, tau_adapt_ms=args.adapt_ms,
                           relative=not args.absolute),
    )
    groups = {**ann.groups(VISUAL_GROUPS), **ann.groups(DN_GROUPS)}
    readout = MotorReadout(groups, brain.device, tau_ms=50.0)
    loco = Locomotion()
    print(f"eye: {eye.n_columns} columns, {int(eye.sees_ground.sum())} see the ground; "
          f"driving {len(drive.idx)} visual neurons")

    frame_ms = 1000.0 / args.fps
    steps = int(round(frame_ms / brain.p.dt))
    n_frames = int(args.seconds * args.fps)
    log = []
    t_wall = time.perf_counter()
    for f in range(n_frames):
        t = f / args.fps
        enemy = None
        if not args.no_enemy:
            dist = max(args.start - args.speed * t, 0.3)
            a = np.radians(args.azimuth)
            enemy = (dist * np.cos(a), dist * np.sin(a))
        crop = render(eye.p.crop_px, eye.p.px_per_unit, enemy, args.radius)
        drive.apply(eye.encode(crop, loco.heading_deg), frame_ms)
        counts = brain.run_steps(steps, readout.idx)
        rates = readout.update(counts, frame_ms)
        heading, speed = loco.update(rates, frame_ms)
        log.append((t, dist if enemy else np.nan, rates, heading, speed))
        if f % max(1, int(args.fps * args.every)) == 0:
            keys = ["L2_R", "Tm1Tm2_R", "T4T5_R", "LC4_R", "LPLC2_R", "LC4_L", "GF", "DNp02",
                    "DNp04", "DNa02_L", "DNa02_R", "DNp09", "MDN"]
            print(f"t={t:4.2f}s d={log[-1][1]:4.1f} "
                  + " ".join(f"{k}={rates.get(k, 0):5.1f}" for k in keys)
                  + f" | head={heading:5.1f} speed={speed:+.2f}")
    wall = time.perf_counter() - t_wall
    print(f"{args.seconds:.1f} s simulated in {wall:.1f} s ({args.seconds / wall:.2f}x realtime)")
    peaks = {k: max(r[2].get(k, 0) for r in log) for k in groups}
    print("peak rates: " + " ".join(f"{k}={v:.1f}" for k, v in peaks.items()))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ts = [r[0] for r in log]
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for ax, keys in zip(axes, (["L2_R", "Tm1Tm2_R", "Mi1_R", "T4T5_R", "T4T5_L"],
                                   ["LC4_R", "LC4_L", "LPLC2_R", "LPLC2_L", "LC6_R"],
                                   ["GF", "DNp02", "DNp04", "DNp06", "DNa02_L", "DNa02_R", "DNp09", "MDN"])):
            for k in keys:
                ax.plot(ts, [r[2].get(k, 0) for r in log], label=k)
            ax.legend(fontsize=7, ncol=4)
            ax.set_ylabel("Hz / neuron")
        axes[-1].set_xlabel("s")
        axes[0].set_title(f"looming from azimuth {args.azimuth} deg" if not args.no_enemy else "no enemy")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
