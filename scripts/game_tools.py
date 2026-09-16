"""Small helpers to drive the game's menus and look at it without playing.

    python scripts/game_tools.py shot out.png            # screenshot of the game window
    python scripts/game_tools.py keys enter down enter   # focus the game and tap keys
    python scripts/game_tools.py info                    # window rectangle, foreground?
"""

from __future__ import annotations

import sys
import time

import numpy as np

from flysurvivors.game import GameWindow, tap_key


def main() -> None:
    cmd = sys.argv[1]
    window = GameWindow()
    if cmd == "info":
        r = window.client_rect()
        print(f"hwnd={window.hwnd} client={r.width}x{r.height} at ({r.left},{r.top}) "
              f"foreground={window.is_foreground()}")
    elif cmd == "shot":
        import cv2
        import dxcam

        r = window.client_rect()
        cam = dxcam.create(output_color="BGR")
        frame = cam.grab(region=(r.left, r.top, r.right, r.bottom))
        if frame is None:
            window.focus()
            time.sleep(0.3)
            frame = cam.grab(region=(r.left, r.top, r.right, r.bottom))
        if frame is None:
            raise SystemExit("no frame (window minimised or region off screen?)")
        out = sys.argv[2]
        scale = min(1.0, 1280 / frame.shape[1])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(out, frame)
        print("saved", out, frame.shape, "mean gray", float(np.mean(frame)))
    elif cmd == "keys":
        window.focus()
        time.sleep(0.4)
        for k in sys.argv[2:]:
            wait = 0.5
            if ":" in k:
                k, wait = k.split(":")
                wait = float(wait)
            tap_key(k)
            time.sleep(wait)
        print("done")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
