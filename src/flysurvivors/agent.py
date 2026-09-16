"""The fly agent: one object that owns brain, eye, drive, readout and locomotion.

Both the pygame arena and the real game feed it player-centred grayscale crops and
read back a heading and a speed. The brain is advanced by the wall-clock time that
passed since the previous tick (capped), so it stays in sync with the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .annotations import Annotations
from .columns import EyeColumns, load_eye_columns
from .connectome import load_connectome
from .eye import DriveParams, EyeParams, RetinaDrive, VirtualEye
from .lif import LIFBrain
from .motor import DN_GROUPS, VISUAL_GROUPS, Locomotion, LocomotionParams, MotorReadout
from .touch import ProximitySense, TouchParams


@dataclass
class AgentParams:
    eye: EyeParams = field(default_factory=EyeParams)
    drive: DriveParams = field(default_factory=lambda: DriveParams(rate_max_hz=250.0))
    locomotion: LocomotionParams = field(default_factory=LocomotionParams)
    touch: TouchParams | None = field(default_factory=TouchParams)  # None disables the ring
    readout_tau_ms: float = 50.0
    max_tick_ms: float = 50.0  # never simulate more than this per tick (keeps latency bounded)
    min_syn: int = 1


class FlyAgent:
    def __init__(self, params: AgentParams | None = None, record_all: bool = False) -> None:
        self.p = params or AgentParams()
        self.cn = load_connectome(min_syn=self.p.min_syn)
        self.ann = Annotations(self.cn)
        self.columns: EyeColumns = load_eye_columns(self.cn)
        self.brain = LIFBrain(self.cn)
        self.eye = VirtualEye(self.columns, self.p.eye)
        self.drive = RetinaDrive(
            self.brain, self.columns, params=self.p.drive, active_columns=self.eye.sees_ground
        )
        self.touch = None
        if self.p.touch is not None:
            self.touch = ProximitySense(self.brain, self.ann, self.crop_px, self.p.eye.px_per_unit,
                                        self.p.touch)
        groups = {**self.ann.groups(VISUAL_GROUPS), **self.ann.groups(DN_GROUPS)}
        self.readout = MotorReadout(groups, self.brain.device, tau_ms=self.p.readout_tau_ms)
        self.loco = Locomotion(self.p.locomotion)
        self.record_all = record_all
        self.last_counts: np.ndarray | None = None  # (N,) spikes in the last tick, if recording
        self.last_intensity: np.ndarray | None = None
        self.last_fly_frame: np.ndarray | None = None
        self.rates: dict[str, float] = {}
        self.sim_ms = 0.0

    @property
    def crop_px(self) -> int:
        return self.p.eye.crop_px

    def tick(self, crop: np.ndarray, dt_ms: float) -> tuple[float, float]:
        """Feed one player-centred grayscale crop (values in [0, 1]) and advance ``dt_ms``.

        Returns ``(heading_deg, speed)``; the stick vector is :meth:`stick`.
        """
        dt_ms = float(min(max(dt_ms, self.brain.p.dt), self.p.max_tick_ms))
        fly_frame = self.eye.to_fly_frame(np.asarray(crop, dtype=np.float32), self.loco.heading_deg)
        self.last_fly_frame = fly_frame
        self.last_intensity = self.eye.encode_fly_frame(fly_frame)
        self.drive.apply(self.last_intensity, dt_ms)
        if self.touch is not None:
            self.touch.apply(fly_frame)
        steps = int(round(dt_ms / self.brain.p.dt))
        if self.record_all:
            counts = self.brain.run_steps(steps)
            self.last_counts = counts.cpu().numpy()
            sub = counts.index_select(0, self.readout.idx)
        else:
            sub = self.brain.run_steps(steps, self.readout.idx)
        self.rates = self.readout.update(sub, dt_ms)
        self.sim_ms += steps * self.brain.p.dt
        return self.loco.update(self.rates, dt_ms)

    def stick(self) -> tuple[float, float]:
        return self.loco.stick()

    def reset(self) -> None:
        self.brain.reset()
        self.readout.reset()
        self.drive.clear()
        if self.touch is not None:
            self.touch.clear()
        self.loco.heading_deg = 0.0
        self.loco.speed = 0.0
        torch.cuda.synchronize() if self.brain.device.type == "cuda" else None
