"""A tiny Vampire-Survivors-like arena to tune and test the fly without the real game.

Top-down, the player at the centre of the view. Dark enemies spawn at the edge and walk
towards the player; the player's aura kills enemies that stay inside it; enemies that
touch the player deal damage. The world is rendered straight into a grayscale crop with
the same pixels-per-unit as the fly's eye, so the agent sees exactly what it would see
from the real game after capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArenaParams:
    crop_px: int = 128
    px_per_unit: float = 16.0
    ground: float = 0.7
    enemy_shade: float = 0.05
    player_shade: float = 0.95
    enemy_radius: float = 0.35
    player_radius: float = 0.3
    player_speed: float = 3.0  # units/s at full stick
    enemy_speed: float = 1.2
    spawn_radius: float = 5.0
    spawn_per_s: float = 2.0
    max_enemies: int = 60
    aura_radius: float = 1.0
    aura_dps: float = 1.0  # enemy hp is 1: dies after 1 s inside the aura
    enemy_dps: float = 0.5
    hp: float = 3.0
    seed: int = 0


class Arena:
    def __init__(self, params: ArenaParams | None = None) -> None:
        self.p = params or ArenaParams()
        self.rng = np.random.default_rng(self.p.seed)
        self.reset()

    def reset(self) -> None:
        self.player = np.zeros(2)
        self.enemies = np.zeros((0, 2))
        self.enemy_hp = np.zeros(0)
        self.hp = self.p.hp
        self.t = 0.0
        self.kills = 0
        self._spawn_acc = 0.0
        self.alive = True

    # --------------------------------------------------------------------- dynamics
    def step(self, stick: tuple[float, float], dt: float) -> None:
        p = self.p
        if not self.alive:
            return
        v = np.array(stick, dtype=float)
        n = np.linalg.norm(v)
        if n > 1:
            v /= n
        self.player = self.player + v * p.player_speed * dt

        self._spawn_acc += p.spawn_per_s * dt
        while self._spawn_acc >= 1 and len(self.enemies) < p.max_enemies:
            self._spawn_acc -= 1
            a = self.rng.uniform(0, 2 * np.pi)
            pos = self.player + p.spawn_radius * np.array([np.cos(a), np.sin(a)])
            self.enemies = np.vstack([self.enemies, pos])
            self.enemy_hp = np.append(self.enemy_hp, 1.0)

        if len(self.enemies):
            d = self.player - self.enemies
            dist = np.linalg.norm(d, axis=1) + 1e-9
            self.enemies = self.enemies + d / dist[:, None] * p.enemy_speed * dt
            dist = np.linalg.norm(self.player - self.enemies, axis=1)
            inside = dist < p.aura_radius
            self.enemy_hp[inside] -= p.aura_dps * dt
            touching = dist < p.enemy_radius + p.player_radius
            self.hp -= p.enemy_dps * dt * touching.sum()
            dead = self.enemy_hp <= 0
            self.kills += int(dead.sum())
            self.enemies = self.enemies[~dead]
            self.enemy_hp = self.enemy_hp[~dead]
        self.t += dt
        if self.hp <= 0:
            self.alive = False

    # ---------------------------------------------------------------------- render
    def render(self) -> np.ndarray:
        """Player-centred grayscale crop, screen convention (x right, y down)."""
        p = self.p
        S = p.crop_px
        img = np.full((S, S), p.ground, dtype=np.float32)
        c = (S - 1) / 2
        yy, xx = np.mgrid[0:S, 0:S]
        for e in self.enemies:
            ex, ey = c + (e[0] - self.player[0]) * p.px_per_unit, c + (e[1] - self.player[1]) * p.px_per_unit
            if -8 < ex < S + 8 and -8 < ey < S + 8:
                img[(xx - ex) ** 2 + (yy - ey) ** 2 <= (p.enemy_radius * p.px_per_unit) ** 2] = p.enemy_shade
        img[(xx - c) ** 2 + (yy - c) ** 2 <= (p.player_radius * p.px_per_unit) ** 2] = p.player_shade
        return img

    def nearest_enemy(self) -> float:
        if not len(self.enemies):
            return np.inf
        return float(np.linalg.norm(self.player - self.enemies, axis=1).min())
