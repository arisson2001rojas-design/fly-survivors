"""Motor readout: descending neuron firing rates -> heading and speed.

Descending neurons (DNs) carry the brain's commands to the ventral nerve cord. A few
are well characterised in walking flies:

- DNa02 (one per side): ipsilateral turning. Right DNa02 active -> turn right.
- DNp09 (one per side): forward walking.
- MDN (two per side, "moonwalker"): backward walking.
- DNp01, the giant fiber (one per side): escape jump, driven by looming detectors
  (LC4, LPLC2). DNp02 / DNp04 / DNp06 also receive looming input.

:class:`MotorReadout` turns spike counts of chosen neuron groups into smoothed rates;
:class:`Locomotion` turns those rates into a heading change and a speed for the game.
Gains are deliberately exposed: they will need tuning against real gameplay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .annotations import Annotations

# Groups used by default: {name: (cell_type, side)}.
DN_GROUPS: dict[str, tuple[str, str | None]] = {
    "DNa02_L": ("DNa02", "left"),
    "DNa02_R": ("DNa02", "right"),
    "DNa01_L": ("DNa01", "left"),
    "DNa01_R": ("DNa01", "right"),
    "DNp09": ("DNp09", None),
    "MDN": ("MDN", None),
    "GF": ("DNp01", None),
    "DNp02": ("DNp02", None),
    "DNp04": ("DNp04", None),
    "DNp06": ("DNp06", None),
}

# Looming / visual projection groups, useful to watch while tuning the eye.
VISUAL_GROUPS: dict[str, tuple[str, str | None]] = {
    "LC4_L": ("LC4", "left"),
    "LC4_R": ("LC4", "right"),
    "LPLC2_L": ("LPLC2", "left"),
    "LPLC2_R": ("LPLC2", "right"),
    "LC6_L": ("LC6", "left"),
    "LC6_R": ("LC6", "right"),
    "T4T5_L": (["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"], "left"),
    "T4T5_R": (["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"], "right"),
    "L1_L": ("L1", "left"),
    "L1_R": ("L1", "right"),
    "L2_R": ("L2", "right"),
    "Tm1Tm2_R": (["Tm1", "Tm2"], "right"),
    "Mi1_R": ("Mi1", "right"),
}


class MotorReadout:
    """Smoothed per-group mean firing rate (Hz per neuron) from spike counts.

    Call :meth:`update` with the spike counts accumulated over ``window_ms`` for the
    neurons in :attr:`idx` (in that order).
    """

    def __init__(self, groups: dict[str, np.ndarray], device, tau_ms: float = 50.0) -> None:
        self.names = list(groups)
        sizes = [len(groups[n]) for n in self.names]
        self.idx = torch.as_tensor(np.concatenate([groups[n] for n in self.names]), device=device)
        member = torch.zeros(len(self.names), len(self.idx), device=device)
        start = 0
        for k, s in enumerate(sizes):
            member[k, start : start + s] = 1.0 / s
            start += s
        self.member = member
        self.tau_ms = tau_ms
        self.rate = torch.zeros(len(self.names), device=device)

    @classmethod
    def from_annotations(
        cls, ann: Annotations, device, spec: dict | None = None, tau_ms: float = 50.0
    ) -> "MotorReadout":
        return cls(ann.groups(spec or DN_GROUPS), device, tau_ms)

    def update(self, counts: torch.Tensor, window_ms: float) -> dict[str, float]:
        """``counts``: (len(idx),) spikes over the window. Returns smoothed rates in Hz."""
        inst = (self.member @ counts.to(torch.float32)) * (1000.0 / window_ms)
        a = math.exp(-window_ms / self.tau_ms)
        self.rate.mul_(a).add_(inst, alpha=1.0 - a)
        return dict(zip(self.names, self.rate.tolist()))

    def reset(self) -> None:
        self.rate.zero_()


@dataclass
class LocomotionParams:
    turn_gain: float = 4.0  # deg/s of heading change per Hz of (DNa02_R - DNa02_L)
    max_turn_deg_s: float = 360.0  # cap on the turning rate
    baseline_speed: float = 0.0  # the fly keeps walking at this speed; the brain steers
    forward_gain: float = 0.02  # speed units per Hz of DNp09
    backward_gain: float = 0.02  # speed units per Hz of MDN
    escape_threshold: float = 5.0  # Hz on the giant fiber
    escape_speed: float = 1.0  # speed during an escape burst
    escape_ms: float = 150.0
    max_speed: float = 1.0


@dataclass
class Locomotion:
    """Integrates DN rates into a heading (deg, screen convention: 0 = +x, clockwise
    positive because screen y points down) and a signed speed along that heading."""

    params: LocomotionParams = field(default_factory=LocomotionParams)
    heading_deg: float = 0.0
    speed: float = 0.0
    _escape_left_ms: float = 0.0

    def update(self, rates: dict[str, float], dt_ms: float) -> tuple[float, float]:
        p = self.params
        turn = (rates.get("DNa02_R", 0.0) - rates.get("DNa02_L", 0.0)) * p.turn_gain
        turn = float(np.clip(turn, -p.max_turn_deg_s, p.max_turn_deg_s))
        self.heading_deg = (self.heading_deg + turn * dt_ms / 1000.0) % 360.0

        if rates.get("GF", 0.0) > p.escape_threshold and self._escape_left_ms <= 0:
            self._escape_left_ms = p.escape_ms
        if self._escape_left_ms > 0:
            self._escape_left_ms -= dt_ms
            self.speed = p.escape_speed  # jump: a dash along the current heading
        else:
            fwd = p.baseline_speed + rates.get("DNp09", 0.0) * p.forward_gain
            back = rates.get("MDN", 0.0) * p.backward_gain
            self.speed = float(np.clip(fwd - back, -p.max_speed, p.max_speed))
        return self.heading_deg, self.speed

    def stick(self) -> tuple[float, float]:
        """Gamepad left-stick vector (x, y) in screen convention, magnitude <= 1."""
        a = math.radians(self.heading_deg)
        return self.speed * math.cos(a), self.speed * math.sin(a)
