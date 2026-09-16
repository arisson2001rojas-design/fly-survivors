"""Let the fly play the built-in arena. Headless by default; ``--display`` opens a window
with the arena, the eye, the DN rates and the brain heat map; ``--video out.mp4`` records it.

    python scripts/play_arena.py --seconds 60 --display
    python scripts/play_arena.py --seconds 60 --policy still     # baseline: do nothing
    python scripts/play_arena.py --seconds 60 --policy random    # baseline: random walk
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from flysurvivors.agent import AgentParams, FlyAgent
from flysurvivors.arena import Arena, ArenaParams
from flysurvivors.motor import LocomotionParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--policy", default="fly", choices=["fly", "still", "random"])
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--video", default=None, help="record the window to an mp4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--turn-gain", type=float, default=4.0)
    ap.add_argument("--forward-gain", type=float, default=0.02)
    ap.add_argument("--escape-threshold", type=float, default=5.0)
    ap.add_argument("--baseline-speed", type=float, default=0.0)
    ap.add_argument("--no-touch", action="store_true", help="disable the bristle proximity ring")
    ap.add_argument("--touch-rate", type=float, default=200.0)
    ap.add_argument("--touch-outer", type=float, default=1.0)
    ap.add_argument("--realtime", action="store_true", help="pace to wall clock")
    ap.add_argument("--log-every", type=float, default=10.0, help="seconds between log lines")
    ap.add_argument("--enemy-radius", type=float, default=0.35)
    ap.add_argument("--spawn-per-s", type=float, default=2.0)
    args = ap.parse_args()

    arena = Arena(ArenaParams(seed=args.seed, enemy_radius=args.enemy_radius, spawn_per_s=args.spawn_per_s))
    dt = 1.0 / args.fps
    agent = None
    if args.policy == "fly":
        params = AgentParams(
            locomotion=LocomotionParams(turn_gain=args.turn_gain, forward_gain=args.forward_gain,
                                        escape_threshold=args.escape_threshold,
                                        baseline_speed=args.baseline_speed),
        )
        params.drive.rate_max_hz = args.rate
        if args.no_touch:
            params.touch = None
        else:
            params.touch.rate_max_hz = args.touch_rate
            params.touch.outer_units = args.touch_outer
        agent = FlyAgent(params, record_all=args.display or bool(args.video))
    rng = np.random.default_rng(args.seed)

    screen = dash = writer = None
    if args.display or args.video:
        import pygame

        from flysurvivors.viz import Dashboard

        pygame.init()
        dash = Dashboard(agent, brain=True) if agent else None
        W = 512 + (dash.size[0] if dash else 0) + 20
        H = max(512, dash.size[1] if dash else 0) + 40
        screen = pygame.display.set_mode((W, H)) if args.display else pygame.Surface((W, H))
        pygame.display.set_caption("fly-survivors arena")
        font = pygame.font.SysFont("consolas", 16)
        if args.video:
            import cv2

            writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    n_frames = int(args.seconds * args.fps)
    t0 = time.perf_counter()
    stick = (0.0, 0.0)
    heading = 0.0
    for f in range(n_frames):
        crop = arena.render()
        if args.policy == "fly":
            agent.tick(crop, dt * 1000.0)
            stick = agent.stick()
        elif args.policy == "random":
            if f % int(args.fps) == 0:
                heading = rng.uniform(0, 360)
            stick = (np.cos(np.radians(heading)), np.sin(np.radians(heading)))
        arena.step(stick, dt)

        if screen is not None:
            import pygame

            if args.display:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        n_frames = f
            screen.fill((15, 15, 20))
            view = pygame.surfarray.make_surface(np.repeat((crop.T * 255).astype(np.uint8)[..., None], 3, axis=2))
            screen.blit(pygame.transform.scale(view, (512, 512)), (10, 30))
            if dash:
                dash.draw(screen, 532, 30)
            hud = f"t={arena.t:5.1f}s  hp={arena.hp:4.1f}  enemies={len(arena.enemies):2d}  kills={arena.kills}"
            screen.blit(font.render(hud, True, (230, 230, 230)), (10, 8))
            if args.display:
                pygame.display.flip()
            if writer is not None:
                import cv2

                frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if args.realtime:
            target = t0 + (f + 1) * dt
            while time.perf_counter() < target:
                pass
        if not arena.alive:
            break
        if f % int(args.log_every * args.fps) == 0 and f:
            r = agent.rates if agent else {}
            print(f"t={arena.t:5.1f}s hp={arena.hp:4.1f} enemies={len(arena.enemies):2d} kills={arena.kills} "
                  f"nearest={arena.nearest_enemy():.2f} | LC4 L/R={r.get('LC4_L', 0):4.1f}/{r.get('LC4_R', 0):4.1f} "
                  f"GF={r.get('GF', 0):4.1f} DNa02 L/R={r.get('DNa02_L', 0):4.1f}/{r.get('DNa02_R', 0):4.1f} "
                  f"touch={np.round(agent.touch.activation, 1).tolist() if agent and agent.touch else ''} "
                  f"head={agent.loco.heading_deg if agent else 0:5.1f} speed={stick[0]:+.2f},{stick[1]:+.2f}")

    wall = time.perf_counter() - t0
    status = "survived" if arena.alive else "died"
    print(f"{args.policy}: {status} at t={arena.t:.1f}s, kills={arena.kills}, hp={max(arena.hp, 0):.1f} "
          f"({arena.t / wall:.2f}x realtime)")
    if writer is not None:
        writer.release()
        print("saved", args.video)


if __name__ == "__main__":
    main()
