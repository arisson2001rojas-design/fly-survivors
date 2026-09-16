"""Live views for pygame windows: the eye, the descending-neuron rates, and the brain.

Everything is drawn from numpy arrays into pygame surfaces; no per-neuron draw calls.
The brain view bins the 138k neurons by their FAFB x/y position into a small image
once, then adds each tick's spike counts to a decaying heat map.
"""

from __future__ import annotations

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None

from .agent import FlyAgent

_HEAT = np.array(
    [[0, 0, 0], [30, 20, 60], [90, 30, 120], [200, 60, 90], [255, 160, 40], [255, 255, 200]],
    dtype=np.float32,
)


def colormap(v: np.ndarray) -> np.ndarray:
    """(H, W) in [0, 1] -> (H, W, 3) uint8 through a dark-to-bright heat palette."""
    v = np.clip(v, 0, 1) * (len(_HEAT) - 1)
    i = np.floor(v).astype(int)
    f = (v - i)[..., None]
    i1 = np.minimum(i + 1, len(_HEAT) - 1)
    return (_HEAT[i] * (1 - f) + _HEAT[i1] * f).astype(np.uint8)


class EyeView:
    """Both eyes unrolled: azimuth left-to-right (-180..180), elevation bottom-to-top."""

    def __init__(self, agent: FlyAgent, width: int = 360, height: int = 160) -> None:
        cols = agent.columns.columns
        self.w, self.h = width, height
        az = cols["az_deg"].to_numpy()
        el = cols["el_deg"].to_numpy()
        self.x = ((az + 180) / 360 * (width - 1)).astype(int)
        self.y = ((90 - el) / 180 * (height - 1)).astype(int)
        self.ground = agent.eye.sees_ground
        self.agent = agent

    def draw(self, surf, x0: int, y0: int) -> None:
        img = np.full((self.h, self.w, 3), 25, dtype=np.uint8)
        inten = self.agent.last_intensity
        if inten is None:
            inten = np.zeros(len(self.x), dtype=np.float32)
        col = np.zeros((len(self.x), 3), dtype=np.uint8)
        v = (np.clip(inten, 0, 1) * 255).astype(np.uint8)
        col[:, 0] = v
        col[:, 1] = v
        col[:, 2] = v
        col[~self.ground] = (40, 40, 70)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xs = np.clip(self.x + dx, 0, self.w - 1)
                ys = np.clip(self.y + dy, 0, self.h - 1)
                img[ys, xs] = col
        # Heading marker: azimuth 0 is straight ahead, in the middle.
        img[:, self.w // 2] = (90, 90, 140)
        surf.blit(pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2))), (x0, y0))


class RatesView:
    KEYS = ["LC4_L", "LC4_R", "GF", "DNp04", "DNa02_L", "DNa02_R", "DNp09", "MDN"]
    SCALE = {"GF": 100.0}

    def __init__(self, agent: FlyAgent, width: int = 220) -> None:
        self.agent = agent
        self.w = width
        self.font = pygame.font.SysFont("consolas", 13)

    def draw(self, surf, x0: int, y0: int) -> int:
        y = y0
        for k in self.KEYS:
            r = self.agent.rates.get(k, 0.0)
            scale = self.SCALE.get(k, 60.0)
            frac = min(r / scale, 1.0)
            pygame.draw.rect(surf, (50, 50, 60), (x0 + 70, y + 2, self.w - 70, 12))
            color = (255, 150, 60) if "DN" in k or k == "GF" else (100, 180, 255)
            pygame.draw.rect(surf, color, (x0 + 70, y + 2, int((self.w - 70) * frac), 12))
            surf.blit(self.font.render(f"{k:8s}{r:5.0f}", True, (220, 220, 220)), (x0, y))
            y += 17
        h, s = self.agent.loco.heading_deg, self.agent.loco.speed
        surf.blit(self.font.render(f"head {h:5.0f}  speed {s:+.2f}", True, (220, 220, 220)), (x0, y))
        return y + 17


class BrainView:
    """Top-down heat map of spiking neurons (needs ``FlyAgent(record_all=True)``)."""

    def __init__(self, agent: FlyAgent, width: int = 360, height: int = 220, decay: float = 0.85):
        df = agent.ann.df
        x = df["pos_x"].to_numpy(float)
        y = df["pos_y"].to_numpy(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        self.idx = df.index.to_numpy()[ok]
        x, y = x[ok], y[ok]
        pad = 0.03
        self.px = ((x - x.min()) / (x.max() - x.min()) * (1 - 2 * pad) + pad) * (width - 1)
        self.py = ((y - y.min()) / (y.max() - y.min()) * (1 - 2 * pad) + pad) * (height - 1)
        self.px = self.px.astype(int)
        self.py = self.py.astype(int)
        self.w, self.h = width, height
        self.bg = np.zeros((height, width), dtype=np.float32)
        np.add.at(self.bg, (self.py, self.px), 1.0)
        self.bg = np.log1p(self.bg)
        self.bg /= self.bg.max()
        self.heat = np.zeros_like(self.bg)
        self.decay = decay
        self.agent = agent

    def draw(self, surf, x0: int, y0: int) -> None:
        counts = self.agent.last_counts
        self.heat *= self.decay
        if counts is not None:
            c = counts[self.idx]
            nz = c > 0
            np.add.at(self.heat, (self.py[nz], self.px[nz]), c[nz])
        img = colormap(0.12 * self.bg + np.clip(self.heat / 6.0, 0, 1))
        surf.blit(pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2))), (x0, y0))


class Dashboard:
    """Eye + rates + brain in one panel. ``draw`` returns the panel size."""

    def __init__(self, agent: FlyAgent, brain: bool = True) -> None:
        self.eye = EyeView(agent)
        self.rates = RatesView(agent)
        self.brain = BrainView(agent) if brain and agent.record_all else None
        self.size = (360 + 230, 160 + (220 if self.brain else 0) + 10)

    def draw(self, surf, x0: int = 0, y0: int = 0) -> None:
        self.eye.draw(surf, x0, y0)
        self.rates.draw(surf, x0 + 370, y0)
        if self.brain:
            self.brain.draw(surf, x0, y0 + 170)
