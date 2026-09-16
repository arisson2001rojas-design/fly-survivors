"""Proximity sense: a ring around the fly, wired to its eye bristles.

Between the ommatidia of the compound eye sit hundreds of mechanosensory bristles
(BM_InOm, ~560 per eye). Probing the connectome shows that activating the right eye's
bristles drives the left DNa02 turning neuron: touch on one side, turn away. That is the
reflex we want when an enemy gets close, so the ring works like this:

- the ring (an annulus of ground around the fly, in world units) is split into sectors,
  each covering a range of azimuth on one side;
- each sector is assigned the bristles of that eye whose position along the
  antero-posterior axis falls in the matching band (front sectors get anterior bristles);
- every frame, the fraction of dark pixels in a sector's annulus becomes a Poisson rate
  on that sector's bristles.

The brain still decides what to do with it. Vision (eye.py) and touch run together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .annotations import Annotations


@dataclass
class TouchParams:
    inner_units: float = 0.35  # annulus inner radius, world units (player is ~1 unit)
    outer_units: float = 1.0  # annulus outer radius: tight, so only the nearest enemy counts
    sectors_per_side: int = 3  # front / middle / back per eye
    rate_max_hz: float = 200.0
    dark_k_std: float = 1.5  # a pixel is "dark" if below the scene mean by this many std
    min_std: float = 0.03
    saturation: float = 0.15  # fraction of dark pixels in a sector that gives full drive
    cell_type: str = "BM_InOm"


class ProximitySense:
    def __init__(self, brain, ann: Annotations, crop_px: int, px_per_unit: float,
                 params: TouchParams | None = None) -> None:
        self.p = params or TouchParams()
        self.brain = brain
        p = self.p
        n = p.sectors_per_side
        df = ann.df

        # Sector -> neuron indices. Sector s on side R covers azimuth [s, s+1) * 180 / n,
        # 0 = straight ahead; anterior bristles have the smallest FAFB z.
        self.sector_idx: list[np.ndarray] = []
        self.sector_az: list[tuple[float, float]] = []
        for side, sign in (("right", 1.0), ("left", -1.0)):
            cells = df[(df["cell_type"] == p.cell_type) & (df["side"] == side)]
            z = cells["pos_z"].to_numpy(float)
            order = np.argsort(z)
            idx = cells.index.to_numpy()[order]
            for s, chunk in enumerate(np.array_split(idx, n)):
                self.sector_idx.append(chunk.astype(np.int64))
                lo, hi = s * 180.0 / n, (s + 1) * 180.0 / n
                self.sector_az.append((sign * lo, sign * hi) if sign > 0 else (-hi, -lo))
        self.n_sectors = len(self.sector_idx)
        all_idx = np.concatenate(self.sector_idx)
        self.idx = torch.as_tensor(all_idx, device=brain.device)
        self.sector_of_cell = np.concatenate(
            [np.full(len(c), k) for k, c in enumerate(self.sector_idx)]
        )
        brain.refr_len[self.idx] = 0

        # Pixel -> sector lookup on the fly-frame crop (heading = +x, y down, az clockwise).
        S = crop_px
        c = (S - 1) / 2
        yy, xx = np.mgrid[0:S, 0:S]
        dist = np.hypot(xx - c, yy - c) / px_per_unit
        az = np.degrees(np.arctan2(yy - c, xx - c))  # -180..180, positive = right side
        ring = (dist >= p.inner_units) & (dist <= p.outer_units)
        self.pixel_sector = np.full((S, S), -1, dtype=np.int64)
        for k, (lo, hi) in enumerate(self.sector_az):
            m = ring & (az >= lo) & (az < hi)
            self.pixel_sector[m] = k
        self.sector_size = np.array([(self.pixel_sector == k).sum() for k in range(self.n_sectors)], float)
        self.activation = np.zeros(self.n_sectors, dtype=np.float32)

    def apply(self, fly_frame: np.ndarray, scene_mean: float | None = None) -> np.ndarray:
        """``fly_frame``: crop already rotated so the heading points to +x, values in [0, 1].
        Returns per-sector activation in [0, 1] and writes the bristle drive to the brain."""
        p = self.p
        mean = float(fly_frame.mean()) if scene_mean is None else scene_mean
        std = max(float(fly_frame.std()), p.min_std)
        dark = fly_frame < mean - p.dark_k_std * std
        counts = np.bincount(self.pixel_sector[dark & (self.pixel_sector >= 0)], minlength=self.n_sectors)
        frac = counts / np.maximum(self.sector_size, 1)
        self.activation = np.clip(frac / p.saturation, 0, 1).astype(np.float32)
        prob = self.activation[self.sector_of_cell] * (p.rate_max_hz * self.brain.p.dt / 1000.0)
        self.brain.stim_prob[self.idx] = torch.as_tensor(prob.astype(np.float32), device=self.brain.device)
        return self.activation

    def clear(self) -> None:
        self.brain.stim_prob[self.idx] = 0.0
        self.activation[:] = 0
