"""Let the fly play the real Vampire Survivors.

Start the game, get into a run (any character, any stage), then:

    python scripts/play_game.py                 # keyboard output, no window
    python scripts/play_game.py --viz           # + a dashboard window (eye, rates, brain)
    python scripts/play_game.py --dry-run       # capture + brain, but never press keys
    python scripts/play_game.py --gamepad       # analog stick through vgamepad (ViGEmBus)

Keys are only sent while the game window is in the foreground. Enter is tapped every
1.5 s so level-up and chest menus pick the highlighted option and the run keeps going.
Ctrl+C stops and releases every key.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from flysurvivors.agent import AgentParams, FlyAgent
from flysurvivors.game import GameWindow, KeyboardOutput, ScreenSource
from flysurvivors.motor import LocomotionParams


def save_debug(out_dir, t, agent, crop, x, y, diff) -> None:
    """Side by side: screen crop, fly-frame crop with the touch ring, eye columns; plus rates."""
    import cv2

    S = crop.shape[0]
    scale = 3
    left = cv2.resize((crop * 255).astype(np.uint8), (S * scale, S * scale), interpolation=cv2.INTER_NEAREST)
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    ff = agent.last_fly_frame if agent.last_fly_frame is not None else crop
    mid = cv2.resize((ff * 255).astype(np.uint8), (S * scale, S * scale), interpolation=cv2.INTER_NEAREST)
    mid = cv2.cvtColor(mid, cv2.COLOR_GRAY2BGR)
    if agent.touch is not None:
        sect = agent.touch.pixel_sector
        act = agent.touch.activation
        overlay = mid.copy()
        for k in range(agent.touch.n_sectors):
            m = cv2.resize((sect == k).astype(np.uint8), (S * scale, S * scale), interpolation=cv2.INTER_NEAREST) > 0
            col = (0, int(255 * act[k]), int(255 * (1 - act[k])))
            overlay[m] = col
        mid = cv2.addWeighted(overlay, 0.35, mid, 0.65, 0)
    # Eye: columns as dots, azimuth left->right, elevation top->bottom.
    eye = np.full((S * scale, S * scale, 3), 30, np.uint8)
    cols = agent.columns.columns
    az = cols["az_deg"].to_numpy()
    el = cols["el_deg"].to_numpy()
    inten = agent.last_intensity if agent.last_intensity is not None else np.zeros(len(az))
    for a, e, v, g in zip(az, el, inten, agent.eye.sees_ground):
        px = int((a + 180) / 360 * (S * scale - 1))
        py = int((90 - e) / 180 * (S * scale - 1))
        c = int(255 * np.clip(v, 0, 1))
        cv2.circle(eye, (px, py), 2, (c, c, c) if g else (80, 40, 40), -1)
    img = np.concatenate([left, mid, eye], axis=1)
    r = agent.rates
    txt = (f"t={t:5.1f}s stick=({x:+.2f},{y:+.2f}) head={agent.loco.heading_deg:5.1f} diff={diff:.4f} | "
           f"LC4 L/R {r.get('LC4_L', 0):.1f}/{r.get('LC4_R', 0):.1f} GF {r.get('GF', 0):.1f} "
           f"DNa02 L/R {r.get('DNa02_L', 0):.1f}/{r.get('DNa02_R', 0):.1f} DNp09 {r.get('DNp09', 0):.1f}")
    if agent.touch is not None:
        txt += " touch " + ",".join(f"{a:.1f}" for a in agent.touch.activation)
    band = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(band, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.imwrite(f"{out_dir}/t{int(t * 10):05d}.png", np.concatenate([band, img], axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Vampire Survivors")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--crop-screen-px", type=int, default=640,
                    help="side of the screen square around the player fed to the eye")
    ap.add_argument("--player-offset", type=int, nargs=2, default=(0, -50), metavar=("DX", "DY"),
                    help="player position relative to the window centre, in screen px")
    ap.add_argument("--focus", action="store_true", help="bring the game window to the front first")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gamepad", action="store_true")
    ap.add_argument("--no-confirm", action="store_true", help="do not tap Enter for menus")
    ap.add_argument("--static-threshold", type=float, default=0.003,
                    help="mean abs frame difference below which the game is considered paused")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--turn-gain", type=float, default=4.0)
    ap.add_argument("--forward-gain", type=float, default=0.02)
    ap.add_argument("--escape-threshold", type=float, default=5.0)
    ap.add_argument("--baseline-speed", type=float, default=0.0)
    ap.add_argument("--no-touch", action="store_true", help="disable the bristle proximity ring")
    ap.add_argument("--touch-rate", type=float, default=200.0)
    ap.add_argument("--touch-outer", type=float, default=1.0)
    ap.add_argument("--video", default=None, help="record the dashboard to an mp4 (with --viz)")
    ap.add_argument("--debug-dir", default=None, help="save what the fly sees every --debug-every s")
    ap.add_argument("--debug-every", type=float, default=2.0)
    args = ap.parse_args()

    window = GameWindow(args.title)
    rect = window.client_rect()
    print(f"game window: {rect.width}x{rect.height} at ({rect.left}, {rect.top})")
    if args.focus:
        window.focus()
        time.sleep(0.3)

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
    agent = FlyAgent(params, record_all=args.viz)
    source = ScreenSource(window, agent.crop_px, args.crop_screen_px, fps=int(args.fps),
                          offset=tuple(args.player_offset))

    if args.gamepad:
        from flysurvivors.game import GamepadOutput

        out = GamepadOutput()
    else:
        out = KeyboardOutput(window)

    screen = dash = writer = None
    if args.viz:
        import pygame

        from flysurvivors.viz import Dashboard

        pygame.init()
        dash = Dashboard(agent, brain=True)
        W, H = 256 + dash.size[0] + 30, max(256, dash.size[1]) + 40
        screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("fly-survivors")
        font = pygame.font.SysFont("consolas", 16)
        if args.video:
            import cv2

            writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    print("running; Ctrl+C to stop" + (" (dry run, no keys)" if args.dry_run else ""))
    period = 1.0 / args.fps
    t_start = last = time.perf_counter()
    frames = 0
    prev_crop = None
    confirms = 0
    next_debug = 0.0
    if args.debug_dir:
        import os

        os.makedirs(args.debug_dir, exist_ok=True)
    try:
        while True:
            now = time.perf_counter()
            if args.seconds and now - t_start > args.seconds:
                break
            crop = source.grab()
            if crop is None:
                time.sleep(0.005)
                continue
            dt_ms = (now - last) * 1000.0
            last = now
            diff = float(np.abs(crop - prev_crop).mean()) if prev_crop is not None else 1.0
            prev_crop = crop
            agent.tick(crop, dt_ms)
            x, y = agent.stick()
            if not args.dry_run:
                out.set_stick(x, y)
                if not args.no_confirm and out.maybe_confirm(diff < args.static_threshold):
                    confirms += 1
                    print(f"static frame (diff={diff:.4f}): tapped Enter")
            frames += 1

            if screen is not None:
                import pygame

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                screen.fill((15, 15, 20))
                view = pygame.surfarray.make_surface(np.repeat((crop.T * 255).astype(np.uint8)[..., None], 3, axis=2))
                screen.blit(pygame.transform.scale(view, (256, 256)), (10, 30))
                dash.draw(screen, 276, 30)
                fg = "" if window.is_foreground() else "  [game not focused: keys off]"
                screen.blit(font.render(f"{frames / (now - t_start + 1e-9):4.1f} fps  stick=({x:+.2f},{y:+.2f}){fg}",
                                        True, (230, 230, 230)), (10, 8))
                pygame.display.flip()
                if writer is not None:
                    import cv2

                    frame = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
                    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if args.debug_dir and now - t_start >= next_debug:
                next_debug += args.debug_every
                save_debug(args.debug_dir, now - t_start, agent, crop, x, y, diff)
            if frames % int(args.fps * 5) == 0:
                r = agent.rates
                print(f"{frames / (now - t_start):4.1f} fps diff={diff:.4f} LC4 L/R={r.get('LC4_L', 0):4.1f}/{r.get('LC4_R', 0):4.1f} "
                      f"GF={r.get('GF', 0):4.1f} DNa02 L/R={r.get('DNa02_L', 0):4.1f}/{r.get('DNa02_R', 0):4.1f} "
                      f"head={agent.loco.heading_deg:5.1f} speed={agent.loco.speed:+.2f}")
            sleep = period - (time.perf_counter() - now)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        out.release_all()
        source.close()
        if writer is not None:
            writer.release()
            print("saved", args.video)
    print(f"stopped after {frames} frames, {agent.sim_ms / 1000:.1f} s of brain time, {confirms} Enter taps")


if __name__ == "__main__":
    main()
