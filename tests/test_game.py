"""Pure-logic parts of the game glue and the arena."""

import numpy as np
import pytest

from flysurvivors.arena import Arena, ArenaParams
from flysurvivors.game import stick_to_keys


def test_stick_to_keys_eight_way():
    assert stick_to_keys(0.0, 0.0) == set()
    assert stick_to_keys(1.0, 0.0) == {"d"}
    assert stick_to_keys(0.0, 1.0) == {"s"}  # screen y down
    assert stick_to_keys(-1.0, 0.0) == {"a"}
    assert stick_to_keys(0.0, -1.0) == {"w"}
    assert stick_to_keys(0.7, 0.7) == {"d", "s"}
    assert stick_to_keys(-0.7, -0.7) == {"a", "w"}
    assert stick_to_keys(0.05, 0.05) == set()  # deadzone


def test_arena_enemies_approach_and_hurt_a_still_player():
    arena = Arena(ArenaParams(seed=1, spawn_per_s=5.0))
    for _ in range(30 * 20):
        arena.step((0.0, 0.0), 1 / 30)
        if not arena.alive:
            break
    assert not arena.alive
    assert arena.kills > 0  # the aura killed some on the way


def test_arena_render_shows_enemies_dark_and_player_bright():
    arena = Arena(ArenaParams(seed=2, spawn_per_s=5.0))
    for _ in range(60):
        arena.step((0.0, 0.0), 1 / 30)
    img = arena.render()
    c = arena.p.crop_px // 2
    assert img[c, c] == pytest.approx(arena.p.player_shade)
    assert (img < 0.1).sum() > 0
    assert img.shape == (arena.p.crop_px, arena.p.crop_px)


def test_arena_player_moves_with_stick():
    arena = Arena(ArenaParams(seed=3, spawn_per_s=0.0))
    for _ in range(30):
        arena.step((1.0, 0.0), 1 / 30)
    assert arena.player[0] == pytest.approx(arena.p.player_speed, rel=1e-3)
    assert arena.player[1] == pytest.approx(0.0)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs CUDA")
def test_agent_reacts_to_dark_object_on_the_right():
    from flysurvivors.agent import FlyAgent

    agent = FlyAgent()
    S, ppu = agent.crop_px, agent.p.eye.px_per_unit
    yy, xx = np.mgrid[0:S, 0:S]
    c = (S - 1) / 2
    quiet = np.full((S, S), 0.7, dtype=np.float32)
    for _ in range(30):
        agent.tick(quiet, 33.0)
    assert agent.rates.get("LC4_R", 0) < 1.0 and agent.rates.get("LC4_L", 0) < 1.0
    # Big dark blob 1.2 units to the fly's right.
    img = quiet.copy()
    img[(xx - c) ** 2 + (yy - (c + 1.2 * ppu)) ** 2 <= (1.2 * ppu) ** 2] = 0.05
    peak_r = peak_l = 0.0
    for _ in range(45):
        agent.tick(img, 33.0)
        peak_r = max(peak_r, agent.rates.get("LC4_R", 0))
        peak_l = max(peak_l, agent.rates.get("LC4_L", 0))
    assert peak_r > 5.0
    assert peak_r > 3 * max(peak_l, 0.1)
