"""Virtual eye: a top-down game frame -> per-column light intensity -> photoreceptor drive.

The fly is imagined standing on the game's ground plane at the player's position, eye
``eye_height`` above it, facing ``heading``. A column looking at azimuth ``az`` and
elevation ``el`` (below the horizon) sees the ground point at distance
``eye_height / tan(-el)`` in direction ``az``. Objects that come closer therefore slide
down the eye and cover more columns: they loom, exactly what the fly's LC4 / LPLC2
looming detectors and the giant fiber respond to. Columns above the horizon see sky.

Each column integrates a Gaussian receptive field of a few degrees; on the ground that
is a blob whose size grows with distance. The whole mapping is a fixed sparse matrix
from crop pixels (fly frame) to columns; per frame we resample the crop into the fly
frame for the current heading and do one sparse mat-vec.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage, sparse

from .columns import EyeColumns


@dataclass
class EyeParams:
    crop_px: int = 128  # side of the square crop centred on the player (fly frame)
    px_per_unit: float = 16.0  # pixels per world unit in the crop
    eye_height: float = 1.0  # world units above the ground plane
    horizon_margin_deg: float = 4.0  # columns closer than this to the horizon see sky
    rf_sigma_deg: float = 2.0  # receptive field Gaussian sigma (angular)
    sky: float = 0.0  # intensity seen above the horizon / outside the crop


class VirtualEye:
    def __init__(self, columns: EyeColumns, params: EyeParams | None = None) -> None:
        self.p = params or EyeParams()
        self.columns = columns
        p = self.p
        az = np.radians(columns.columns["az_deg"].to_numpy(dtype=float))
        el = np.radians(columns.columns["el_deg"].to_numpy(dtype=float))
        n = len(az)
        S = p.crop_px
        c = (S - 1) / 2.0

        ground = el < -np.radians(p.horizon_margin_deg)
        d = np.where(ground, p.eye_height / np.tan(-np.clip(el, None, -1e-3)), np.inf)
        # Fly frame: heading = +x, y down (screen convention), positive az = clockwise = right side.
        px = c + d * p.px_per_unit * np.cos(az)
        py = c + d * p.px_per_unit * np.sin(az)
        sigma = d * p.px_per_unit * np.radians(p.rf_sigma_deg) / np.maximum(np.sin(-el), 1e-3)
        sigma = np.clip(sigma, 0.5, S / 4)

        rows, cols, vals = [], [], []
        yy, xx = np.mgrid[0:S, 0:S]
        for i in range(n):
            if not ground[i] or not (-3 * sigma[i] < px[i] < S + 3 * sigma[i]) or not (
                -3 * sigma[i] < py[i] < S + 3 * sigma[i]
            ):
                continue
            x0, x1 = int(max(0, px[i] - 3 * sigma[i])), int(min(S, px[i] + 3 * sigma[i] + 1))
            y0, y1 = int(max(0, py[i] - 3 * sigma[i])), int(min(S, py[i] + 3 * sigma[i] + 1))
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = xx[y0:y1, x0:x1], yy[y0:y1, x0:x1]
            w = np.exp(-((gx - px[i]) ** 2 + (gy - py[i]) ** 2) / (2 * sigma[i] ** 2))
            tot = w.sum()
            if tot < 1e-6:
                continue
            rows.append(np.full(w.size, i))
            cols.append((gy * S + gx).ravel())
            vals.append((w / tot).ravel())
        self.M = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, S * S)
        )
        self.sees_ground = np.asarray(self.M.sum(axis=1)).ravel() > 0.5
        self.ground_xy = np.stack([px, py], axis=1)
        self.distance = d
        # Fly-frame sampling grid, rotated per heading in :meth:`to_fly_frame`.
        self._u = xx - c
        self._v = yy - c
        self._c = c

    @property
    def n_columns(self) -> int:
        return self.M.shape[0]

    def to_fly_frame(self, crop: np.ndarray, heading_deg: float) -> np.ndarray:
        """Resample a player-centred screen crop so that the heading points to +x."""
        if crop.shape != (self.p.crop_px, self.p.crop_px):
            raise ValueError(f"crop must be {self.p.crop_px}x{self.p.crop_px}")
        a = np.radians(heading_deg)
        xs = self._c + self._u * np.cos(a) - self._v * np.sin(a)
        ys = self._c + self._u * np.sin(a) + self._v * np.cos(a)
        return ndimage.map_coordinates(crop, [ys, xs], order=1, mode="nearest")

    def encode(self, crop: np.ndarray, heading_deg: float = 0.0) -> np.ndarray:
        """Per-column intensity in [0, 1] from a player-centred grayscale crop in [0, 1]."""
        img = self.to_fly_frame(np.asarray(crop, dtype=np.float32), heading_deg)
        out = self.M @ img.ravel()
        out[~self.sees_ground] = self.p.sky
        return out.astype(np.float32)


@dataclass
class DriveParams:
    rate_max_hz: float = 150.0  # input rate at full drive
    relative: bool = True  # encode contrast relative to the mean of the visible scene
    contrast_scale: float = 0.4  # intensity difference that saturates the drive
    tonic_gain: float = 1.0  # weight of the instantaneous (relative) intensity
    phasic_gain: float = 0.0  # weight of (intensity - slow average): temporal contrast
    baseline: float = 0.0  # added drive, in [0, 1]
    tau_adapt_ms: float = 300.0  # time constant of the slow average


# Which cells to drive and with what polarity: +1 = fires for light, -1 = fires for dark.
#
# Photoreceptors depolarise to light, but their synapses onto the lamina are sparse in
# FAFB and mis-signed in the model (histamine is inhibitory; predicted mostly excitatory),
# and the lamina -> medulla -> T4/T5 cascade dies out in the LIF model (it cannot compute
# motion). What does work, measured with scripts/probe_pathway.py: the medulla OFF cells
# Tm1/Tm2/Tm4/Tm9 drive LC4 (dark-object detectors) which drive DNp02/DNp04/DNp06 and a
# contralateral DNa02 turn. So by default we inject dark contrast there, retinotopically,
# and also into the lamina cells so the earlier stages light up. The real fly's lamina and
# medulla OFF cells do depolarise for dark, hence polarity -1.
DEFAULT_DRIVE_TYPES: dict[str, float] = {
    "L1": -1.0, "L2": -1.0, "L3": -1.0, "Tm1": -1.0, "Tm2": -1.0, "Tm4": -1.0, "Tm9": -1.0,
}


class RetinaDrive:
    """Writes per-column intensities as Poisson rates onto visual input neurons."""

    def __init__(
        self,
        brain,
        columns: EyeColumns,
        cell_types: dict[str, float] | None = None,
        params: DriveParams | None = None,
        active_columns: np.ndarray | None = None,
    ) -> None:
        self.p = params or DriveParams()
        self.brain = brain
        cell_types = cell_types or DEFAULT_DRIVE_TYPES
        idx, rows, pol = [], [], []
        for ct, polarity in cell_types.items():
            i, r = columns.indices(ct)
            idx.append(i)
            rows.append(r)
            pol.append(np.full(len(i), polarity, dtype=np.float32))
        self.rows = np.concatenate(rows)
        self.polarity = np.concatenate(pol)
        self.idx = torch.as_tensor(np.concatenate(idx), device=brain.device)
        n_cols = len(columns.columns)
        self.active = np.ones(n_cols, bool) if active_columns is None else active_columns
        brain.refr_len[self.idx] = 0  # driven like the reference's Poisson inputs
        self.slow: np.ndarray | None = None

    def apply(self, intensity: np.ndarray, dt_ms: float) -> np.ndarray:
        """Update the brain's stimulus probabilities. Returns the per-column drive in [-1, 1]
        (positive = brighter than the adapted level for phasic gain, or plain intensity)."""
        p = self.p
        if p.relative:
            mean = float(intensity[self.active].mean()) if self.active.any() else 0.0
            signal = (intensity - mean) / p.contrast_scale  # >0 brighter, <0 darker
        else:
            signal = intensity * 2 - 1
        if self.slow is None:
            self.slow = signal.astype(np.float32).copy()
        a = np.exp(-dt_ms / p.tau_adapt_ms)
        self.slow = a * self.slow + (1 - a) * signal
        light = p.tonic_gain * signal + p.phasic_gain * (signal - self.slow)
        per_cell = np.where(self.polarity > 0, light[self.rows], -light[self.rows])
        per_cell = np.clip(p.baseline + per_cell, 0, 1) * self.active[self.rows]
        prob = per_cell * (p.rate_max_hz * self.brain.p.dt / 1000.0)
        self.brain.stim_prob[self.idx] = torch.as_tensor(
            prob.astype(np.float32), device=self.brain.device
        )
        return np.where(self.active, intensity, np.nan)

    def clear(self) -> None:
        self.brain.stim_prob[self.idx] = 0.0
        self.slow = None
