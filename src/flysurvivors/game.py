"""Talking to the real game on Windows: screen capture in, keyboard (or gamepad) out.

- :class:`GameWindow` finds the game's window and its client rectangle.
- :class:`ScreenSource` grabs frames with dxcam (DirectX desktop duplication) and returns
  a grayscale crop centred on the player, who sits at the centre of the game view.
- :class:`KeyboardOutput` presses W/A/S/D with scan codes through SendInput, which games
  reading DirectInput accept. Keys are only sent while the game window is in the
  foreground, and all keys are released on exit. 8-way movement is enough for the game.
- :class:`GamepadOutput` is the analog alternative through vgamepad (needs the ViGEmBus
  driver; optional).
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import math
import time
from dataclasses import dataclass

import numpy as np

user32 = ctypes.windll.user32 if hasattr(ctypes, "windll") else None


# ----------------------------------------------------------------------------- window
@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2


class GameWindow:
    def __init__(self, title_contains: str = "Vampire Survivors") -> None:
        self.title = title_contains
        self.hwnd = self._find()

    def _find(self) -> int:
        found = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)
        def cb(hwnd, _):
            if user32.IsWindowVisible(hwnd):
                n = user32.GetWindowTextLengthW(hwnd)
                if n:
                    buf = ctypes.create_unicode_buffer(n + 1)
                    user32.GetWindowTextW(hwnd, buf, n + 1)
                    if self.title.lower() in buf.value.lower():
                        found.append(hwnd)
            return True

        user32.EnumWindows(cb, 0)
        if not found:
            raise RuntimeError(f"no visible window with '{self.title}' in its title")
        return found[0]

    def client_rect(self) -> Rect:
        r = wt.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(r))
        tl = wt.POINT(r.left, r.top)
        br = wt.POINT(r.right, r.bottom)
        user32.ClientToScreen(self.hwnd, ctypes.byref(tl))
        user32.ClientToScreen(self.hwnd, ctypes.byref(br))
        return Rect(tl.x, tl.y, br.x, br.y)

    def is_foreground(self) -> bool:
        return user32.GetForegroundWindow() == self.hwnd

    def focus(self) -> None:
        user32.SetForegroundWindow(self.hwnd)


# ---------------------------------------------------------------------------- capture
class ScreenSource:
    """dxcam-based grabber returning player-centred grayscale crops in [0, 1]."""

    def __init__(self, window: GameWindow, crop_px: int, crop_screen_px: int, fps: int = 60,
                 offset: tuple[int, int] = (0, 0)):
        import cv2
        import dxcam

        self.cv2 = cv2
        self.window = window
        self.crop_px = crop_px
        self.crop_screen_px = crop_screen_px
        self.offset = offset
        self.cam = dxcam.create(output_color="BGR")
        self.region = self._region()
        self.cam.start(region=self.region, target_fps=fps, video_mode=True)
        self.last_bgr: np.ndarray | None = None

    def _region(self) -> tuple[int, int, int, int]:
        r = self.window.client_rect()
        cx, cy = r.center
        cx += self.offset[0]
        cy += self.offset[1]
        h = self.crop_screen_px // 2
        return (cx - h, cy - h, cx + h, cy + h)

    def grab(self) -> np.ndarray | None:
        frame = self.cam.get_latest_frame()
        if frame is None:
            return None
        self.last_bgr = frame
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        small = self.cv2.resize(gray, (self.crop_px, self.crop_px), interpolation=self.cv2.INTER_AREA)
        return small.astype(np.float32) / 255.0

    def close(self) -> None:
        try:
            self.cam.stop()
        except Exception:
            pass


# ----------------------------------------------------------------------------- output
# DirectInput scan codes.
SCAN = {
    "w": 0x11, "a": 0x1E, "s": 0x1F, "d": 0x20, "enter": 0x1C, "space": 0x39, "esc": 0x01,
    "up": 0xC8, "down": 0xD0, "left": 0xCB, "right": 0xCD,
}
EXTENDED = {"up", "down", "left", "right"}
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wt.WORD),
        ("wScan", wt.WORD),
        ("dwFlags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("padding", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTUNION)]


def _send_key(scan: int, up: bool, extended: bool = False) -> None:
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
        scan &= 0xFF
    inp = _INPUT(type=INPUT_KEYBOARD, u=_INPUTUNION(ki=_KEYBDINPUT(0, scan, flags, 0, None)))
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def tap_key(key: str, hold_s: float = 0.08) -> None:
    """Press and release a named key (see SCAN), regardless of focus."""
    _send_key(SCAN[key], up=False, extended=key in EXTENDED)
    time.sleep(hold_s)
    _send_key(SCAN[key], up=True, extended=key in EXTENDED)


def stick_to_keys(x: float, y: float, deadzone: float = 0.15) -> set[str]:
    """Screen-convention stick (x right, y down) -> subset of {w, a, s, d}, 8-way."""
    if math.hypot(x, y) < deadzone:
        return set()
    ang = math.degrees(math.atan2(y, x)) % 360.0  # 0 = right, 90 = down
    sector = int(((ang + 22.5) % 360) // 45)
    return [
        {"d"}, {"d", "s"}, {"s"}, {"s", "a"}, {"a"}, {"a", "w"}, {"w"}, {"w", "d"}
    ][sector]


class KeyboardOutput:
    def __init__(self, window: GameWindow | None = None, confirm_every_s: float = 1.5,
                 static_before_confirm_s: float = 0.8) -> None:
        self.window = window
        self.held: set[str] = set()
        self.confirm_every_s = confirm_every_s
        self.static_before_confirm_s = static_before_confirm_s
        self._last_confirm = 0.0
        self._static_since: float | None = None

    def _allowed(self) -> bool:
        return self.window is None or self.window.is_foreground()

    def set_stick(self, x: float, y: float) -> set[str]:
        want = stick_to_keys(x, y) if self._allowed() else set()
        for k in self.held - want:
            _send_key(SCAN[k], up=True)
        for k in want - self.held:
            _send_key(SCAN[k], up=False)
        self.held = want
        return want

    def tap(self, key: str, hold_s: float = 0.08) -> None:
        if not self._allowed():
            return
        tap_key(key, hold_s)

    def maybe_confirm(self, frame_static: bool) -> bool:
        """Tap Enter when the game has been showing a static frame for a while: level-up,
        chest and game-over screens pause the action; menus pick the highlighted option.
        Never during play, where the pause menu would be opened and 'Quit' could follow."""
        now = time.perf_counter()
        if not frame_static:
            self._static_since = None
            return False
        if self._static_since is None:
            self._static_since = now
            return False
        if now - self._static_since < self.static_before_confirm_s:
            return False
        # The game is paused on a menu: movement keys are useless there and a held
        # direction makes the menu ignore Enter, so release them first.
        self.release_all()
        if now - self._last_confirm >= self.confirm_every_s:
            self._last_confirm = now
            time.sleep(0.05)
            self.tap("enter")
            return True
        return False

    def release_all(self) -> None:
        for k in self.held:
            _send_key(SCAN[k], up=True)
        self.held = set()


class GamepadOutput:
    """Analog left stick through a virtual Xbox 360 pad (pip install vgamepad)."""

    def __init__(self) -> None:
        import vgamepad as vg

        self.vg = vg
        self.pad = vg.VX360Gamepad()
        self._last_confirm = 0.0

    def set_stick(self, x: float, y: float) -> None:
        self.pad.left_joystick_float(x_value_float=float(np.clip(x, -1, 1)),
                                     y_value_float=float(np.clip(-y, -1, 1)))
        self.pad.update()

    def maybe_confirm(self, frame_static: bool, every_s: float = 1.5) -> bool:
        now = time.perf_counter()
        if not frame_static or now - self._last_confirm < every_s:
            return False
        self._last_confirm = now
        self.pad.press_button(self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.pad.update()
        time.sleep(0.03)
        self.pad.release_button(self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.pad.update()
        return True

    def release_all(self) -> None:
        self.pad.reset()
        self.pad.update()
