# TH06C Read-Only Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Windows-only, read-only probe for the verified Steam `th06c.exe` build that can attach safely, verify the executable, resolve `module_base + RVA`, validate known public RVAs, and iteratively discover unknown gameplay values without modifying Touhou.

**Architecture:** Keep Win32 process access isolated behind a small backend so almost all probe logic remains testable on Linux with fake memory. The probe uses exact-build verification before reading known RVAs, exposes a Cheat-Engine-style typed candidate scanner for unknown values, and persists candidate sets between short observation steps. This plan deliberately stops before DLL injection, LAN transport, retinal projection, or brain control; its deliverable is a trustworthy discovery tool for Phase A of the approved architecture.

**Tech Stack:** Python 3.11+, standard-library `ctypes`/`hashlib`/`struct`/`argparse`, NumPy, pytest, Windows 10 x64, Win32 Toolhelp + `OpenProcess` + `ReadProcessMemory` + `VirtualQueryEx`.

**Spec:** `docs/superpowers/specs/2026-09-16-flytouhou-architecture-design.md`

## Global Constraints

- Target executable: `th06c.exe` with SHA-256 `1E2F280EE8EDE3018AABBD72897A46684957EE18442A89C3E2D6194C06B628FD`.
- Runtime addresses are always `module_base + RVA`; never assume the preferred PE image base.
- This subproject is strictly read-only: do not call `WriteProcessMemory`, inject DLLs, patch bytes, send gameplay input, or alter Touhou state.
- Known public RVAs are sanity checks only. Unknown player/bullet/laser/enemy/lives/Continue/stage addresses must not be invented or copied from `th06`, `th06nc.exe`, or another executable.
- The probe must refuse known-RVA operations when the executable hash does not match the verified build.
- Windows-specific imports must not make the package unimportable on Linux; tests use fake process-memory backends.
- No new runtime dependency is required; NumPy and pytest already exist in the project.
- Do not change the FlyWire LIF simulation, retina, motor path, or existing game-control behavior in this plan.
- Keep each discovery artifact local and disposable; do not commit raw process-memory dumps to git.

---

## File Structure

Create these focused units:

- `src/flysurvivors/touhou/__init__.py` — public probe exports only.
- `src/flysurvivors/touhou/builds.py` — verified build metadata, SHA-256 verification, known RVAs, and RVA resolution.
- `src/flysurvivors/touhou/memory.py` — platform-neutral memory interfaces and data types used by scanners.
- `src/flysurvivors/touhou/winproc.py` — Windows-only process discovery, module-base lookup, region enumeration, and read-only memory access.
- `src/flysurvivors/touhou/scanner.py` — typed candidate scanning/rescanning and `.npz` persistence; no Win32 calls.
- `src/flysurvivors/touhou/probe_cli.py` — argument parser and command handlers that compose the modules above.
- `scripts/probe_th06c.py` — tiny executable entry point.
- `tests/test_touhou_builds.py` — build-verification/RVA tests.
- `tests/test_touhou_memory.py` — platform-neutral memory model tests.
- `tests/test_touhou_scanner.py` — candidate scan/rescan/persistence tests.
- `tests/test_touhou_probe_cli.py` — CLI parsing and safety-gate tests with fakes.
- `docs/th06c-probe.md` — exact operator workflow for Windows PowerShell.

---

### Task 1: Verified Build Metadata and RVA Safety Gate

**Files:**
- Create: `src/flysurvivors/touhou/__init__.py`
- Create: `src/flysurvivors/touhou/builds.py`
- Test: `tests/test_touhou_builds.py`

**Interfaces:**
- Produces: `TH06C_CLASSIC`, `TouhouBuild`, `BuildVerification`, `sha256_file(path)`, `verify_build(path, build)`, `resolve_rva(module_base, rva)`.
- Known RVAs exposed by `TH06C_CLASSIC.known_rvas`: `menu_cursor=0x00B5C168`, `score=0x003A3B4C`.

- [ ] **Step 1: Write failing tests for hashing, exact-build verification, mismatch rejection, and RVA resolution**

```python
# tests/test_touhou_builds.py
from pathlib import Path

from flysurvivors.touhou.builds import (
    TH06C_CLASSIC,
    TouhouBuild,
    resolve_rva,
    sha256_file,
    verify_build,
)


def test_sha256_file_and_verification(tmp_path: Path):
    exe = tmp_path / "th06c.exe"
    exe.write_bytes(b"flytouhou-test")
    digest = sha256_file(exe)
    build = TouhouBuild(
        exe_name="th06c.exe",
        sha256=digest,
        known_rvas={"score": 0x1234},
    )
    result = verify_build(exe, build)
    assert result.ok
    assert result.actual_sha256 == digest.upper()


def test_verify_build_rejects_wrong_hash(tmp_path: Path):
    exe = tmp_path / "th06c.exe"
    exe.write_bytes(b"wrong")
    result = verify_build(exe, TH06C_CLASSIC)
    assert not result.ok
    assert result.expected_sha256 == TH06C_CLASSIC.sha256


def test_verified_build_metadata_is_exact():
    assert TH06C_CLASSIC.exe_name == "th06c.exe"
    assert TH06C_CLASSIC.sha256 == "1E2F280EE8EDE3018AABBD72897A46684957EE18442A89C3E2D6194C06B628FD"
    assert TH06C_CLASSIC.known_rvas["menu_cursor"] == 0x00B5C168
    assert TH06C_CLASSIC.known_rvas["score"] == 0x003A3B4C


def test_resolve_rva_uses_runtime_module_base():
    assert resolve_rva(0x7FF600000000, 0x003A3B4C) == 0x7FF6003A3B4C
```

- [ ] **Step 2: Run the tests and verify they fail because the module does not exist**

Run:

```powershell
python -m pytest tests/test_touhou_builds.py -v
```

Expected: collection/import failure for `flysurvivors.touhou.builds`.

- [ ] **Step 3: Implement immutable build metadata and verification**

```python
# src/flysurvivors/touhou/builds.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
import hashlib


@dataclass(frozen=True)
class TouhouBuild:
    exe_name: str
    sha256: str
    known_rvas: Mapping[str, int]


@dataclass(frozen=True)
class BuildVerification:
    ok: bool
    expected_sha256: str
    actual_sha256: str
    path: Path


TH06C_CLASSIC = TouhouBuild(
    exe_name="th06c.exe",
    sha256="1E2F280EE8EDE3018AABBD72897A46684957EE18442A89C3E2D6194C06B628FD",
    known_rvas=MappingProxyType({
        "menu_cursor": 0x00B5C168,
        "score": 0x003A3B4C,
    }),
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def verify_build(path: str | Path, build: TouhouBuild = TH06C_CLASSIC) -> BuildVerification:
    p = Path(path)
    actual = sha256_file(p)
    return BuildVerification(actual == build.sha256, build.sha256, actual, p)


def resolve_rva(module_base: int, rva: int) -> int:
    if module_base < 0 or rva < 0:
        raise ValueError("module_base and rva must be non-negative")
    return module_base + rva
```

Export only the public build symbols from `src/flysurvivors/touhou/__init__.py`.

- [ ] **Step 4: Run the focused tests**

```powershell
python -m pytest tests/test_touhou_builds.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/flysurvivors/touhou tests/test_touhou_builds.py
git commit -m "feat: add verified th06c build metadata"
```

---

### Task 2: Platform-Neutral Read-Only Memory Contract

**Files:**
- Create: `src/flysurvivors/touhou/memory.py`
- Test: `tests/test_touhou_memory.py`

**Interfaces:**
- Produces: `MemoryRegion`, `ProcessMemory` protocol, `MemoryReadError`, `read_u8`, `read_i32`, `read_u32`, `read_f32`, `read_f64`.
- Consumes: no Windows API; later tasks implement the protocol.

- [ ] **Step 1: Write failing tests using a fake backend**

```python
# tests/test_touhou_memory.py
import struct

import pytest

from flysurvivors.touhou.memory import (
    MemoryReadError,
    MemoryRegion,
    read_f32,
    read_i32,
    read_u32,
)


class FakeMemory:
    def __init__(self, base: int, data: bytes):
        self.base = base
        self.data = data

    def read(self, address: int, size: int) -> bytes:
        off = address - self.base
        if off < 0 or off + size > len(self.data):
            raise MemoryReadError(address, size)
        return self.data[off:off + size]


def test_scalar_readers_are_little_endian():
    base = 0x1000
    data = struct.pack("<iIf", -7, 123, 1.25)
    mem = FakeMemory(base, data)
    assert read_i32(mem, base) == -7
    assert read_u32(mem, base + 4) == 123
    assert read_f32(mem, base + 8) == pytest.approx(1.25)


def test_memory_region_end_and_readable_flag():
    r = MemoryRegion(base=0x2000, size=0x1000, readable=True, writable=True, executable=False)
    assert r.end == 0x3000
```

- [ ] **Step 2: Run the tests and confirm failure**

```powershell
python -m pytest tests/test_touhou_memory.py -v
```

Expected: import failure because `memory.py` is absent.

- [ ] **Step 3: Implement the contract and typed readers**

```python
# src/flysurvivors/touhou/memory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol
import struct


class MemoryReadError(RuntimeError):
    def __init__(self, address: int, size: int):
        super().__init__(f"cannot read {size} bytes at 0x{address:X}")
        self.address = address
        self.size = size


@dataclass(frozen=True)
class MemoryRegion:
    base: int
    size: int
    readable: bool
    writable: bool
    executable: bool

    @property
    def end(self) -> int:
        return self.base + self.size


class ProcessMemory(Protocol):
    pid: int
    module_base: int

    def read(self, address: int, size: int) -> bytes: ...
    def regions(self) -> Iterator[MemoryRegion]: ...
    def close(self) -> None: ...


def _unpack(mem: ProcessMemory, address: int, fmt: str):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, mem.read(address, size))[0]


def read_u8(mem: ProcessMemory, address: int) -> int:
    return _unpack(mem, address, "<B")


def read_i32(mem: ProcessMemory, address: int) -> int:
    return _unpack(mem, address, "<i")


def read_u32(mem: ProcessMemory, address: int) -> int:
    return _unpack(mem, address, "<I")


def read_f32(mem: ProcessMemory, address: int) -> float:
    return _unpack(mem, address, "<f")


def read_f64(mem: ProcessMemory, address: int) -> float:
    return _unpack(mem, address, "<d")
```

- [ ] **Step 4: Run the focused tests**

```powershell
python -m pytest tests/test_touhou_memory.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/flysurvivors/touhou/memory.py tests/test_touhou_memory.py
git commit -m "feat: add read-only process memory contract"
```

---

### Task 3: Windows Process Attachment and Memory Reading

**Files:**
- Create: `src/flysurvivors/touhou/winproc.py`
- Modify: `src/flysurvivors/touhou/__init__.py`
- Test: `tests/test_touhou_memory.py`

**Interfaces:**
- Produces: `WindowsProcessMemory.attach(exe_name="th06c.exe")`, `find_process_id(exe_name)`, `find_module_base(pid, module_name)`.
- `WindowsProcessMemory` implements the `ProcessMemory` protocol and opens the process with read/query rights only.

- [ ] **Step 1: Add a cross-platform import-safety test**

```python
# append to tests/test_touhou_memory.py

def test_winproc_module_is_importable_without_attaching():
    import flysurvivors.touhou.winproc as winproc
    assert hasattr(winproc, "WindowsProcessMemory")
```

- [ ] **Step 2: Run it before implementation**

```powershell
python -m pytest tests/test_touhou_memory.py::test_winproc_module_is_importable_without_attaching -v
```

Expected: FAIL because `winproc.py` is absent.

- [ ] **Step 3: Implement the Windows backend with read-only rights**

Use these rights and constants only:

```python
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MEM_COMMIT = 0x1000
PAGE_GUARD = 0x100
PAGE_NOACCESS = 0x01
```

The backend must:

```python
class WindowsProcessMemory:
    pid: int
    module_base: int

    @classmethod
    def attach(cls, exe_name: str = "th06c.exe") -> "WindowsProcessMemory": ...

    def read(self, address: int, size: int) -> bytes: ...
    def regions(self) -> Iterator[MemoryRegion]: ...
    def close(self) -> None: ...
    def __enter__(self) -> "WindowsProcessMemory": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
```

Implementation requirements:

```python
if sys.platform != "win32":
    # importing is allowed; attaching raises a clear RuntimeError.
```

Use `CreateToolhelp32Snapshot` + `Process32FirstW`/`Process32NextW` to find `th06c.exe`, `Module32FirstW`/`Module32NextW` to obtain the runtime image base, `OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, ...)` for the handle, `ReadProcessMemory` for reads, and `VirtualQueryEx` for committed readable regions. Skip `PAGE_GUARD` and `PAGE_NOACCESS` regions. Do not request `PROCESS_VM_WRITE` or `PROCESS_VM_OPERATION`.

On any short/failed read, raise `MemoryReadError(address, size)` rather than returning partial data.

- [ ] **Step 4: Run the package tests on the development machine**

```powershell
python -m pytest tests/test_touhou_memory.py -v
```

Expected: all tests pass without needing Touhou to be running.

- [ ] **Step 5: Perform one manual Windows read-only smoke check**

With Touhou running, execute a temporary one-liner from the repo root:

```powershell
python -c "from flysurvivors.touhou.winproc import WindowsProcessMemory; m=WindowsProcessMemory.attach(); print(m.pid, hex(m.module_base)); m.close()"
```

Expected: prints the PID and a nonzero ASLR module base; Touhou continues running unchanged.

- [ ] **Step 6: Commit**

```powershell
git add src/flysurvivors/touhou/winproc.py src/flysurvivors/touhou/__init__.py tests/test_touhou_memory.py
git commit -m "feat: add Windows read-only process attachment"
```

---

### Task 4: Known-RVA Sanity Probe

**Files:**
- Create: `src/flysurvivors/touhou/probe_cli.py`
- Create: `scripts/probe_th06c.py`
- Test: `tests/test_touhou_probe_cli.py`

**Interfaces:**
- Produces CLI commands `info` and `known`.
- `info --exe PATH` verifies SHA-256 and reports process/module information.
- `known --exe PATH` refuses mismatched builds, then reads `menu_cursor` and `score` through `module_base + RVA`.

- [ ] **Step 1: Write parser and safety-gate tests using dependency injection**

```python
# tests/test_touhou_probe_cli.py
from pathlib import Path

from flysurvivors.touhou.builds import BuildVerification, TH06C_CLASSIC
from flysurvivors.touhou.probe_cli import build_parser, require_verified_build


def test_parser_has_info_and_known_commands():
    parser = build_parser()
    assert parser.parse_args(["info", "--exe", "C:/x/th06c.exe"]).command == "info"
    assert parser.parse_args(["known", "--exe", "C:/x/th06c.exe"]).command == "known"


def test_require_verified_build_rejects_mismatch():
    result = BuildVerification(
        ok=False,
        expected_sha256=TH06C_CLASSIC.sha256,
        actual_sha256="00" * 32,
        path=Path("th06c.exe"),
    )
    try:
        require_verified_build(result)
    except RuntimeError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("mismatched build must be rejected")
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest tests/test_touhou_probe_cli.py -v
```

Expected: import failure because `probe_cli.py` does not exist.

- [ ] **Step 3: Implement the CLI composition layer**

The entry point remains tiny:

```python
# scripts/probe_th06c.py
from flysurvivors.touhou.probe_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

`probe_cli.py` must expose:

```python
def build_parser() -> argparse.ArgumentParser: ...
def require_verified_build(result: BuildVerification) -> None: ...
def cmd_info(args) -> int: ...
def cmd_known(args) -> int: ...
def main(argv: list[str] | None = None) -> int: ...
```

`cmd_known` reads:

```python
score_addr = resolve_rva(mem.module_base, TH06C_CLASSIC.known_rvas["score"])
menu_addr = resolve_rva(mem.module_base, TH06C_CLASSIC.known_rvas["menu_cursor"])
score = read_u32(mem, score_addr)
menu_cursor = read_i32(mem, menu_addr)
```

Before attaching for `known`, call `verify_build(args.exe)` and `require_verified_build(...)`.

- [ ] **Step 4: Run automated tests**

```powershell
python -m pytest tests/test_touhou_probe_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Validate against the real game without changing it**

At the Touhou title/menu screen:

```powershell
python scripts/probe_th06c.py info --exe "C:\Program Files (x86)\Steam\steamapps\common\th06c\th06c.exe"
python scripts/probe_th06c.py known --exe "C:\Program Files (x86)\Steam\steamapps\common\th06c\th06c.exe"
```

Expected:

- SHA reports the exact verified digest.
- PID/module base are nonzero.
- score is readable.
- menu cursor is readable and changes consistently when the human moves through the main menu.

Do not proceed to unknown-memory scanning if these sanity checks are unstable.

- [ ] **Step 6: Commit**

```powershell
git add src/flysurvivors/touhou/probe_cli.py scripts/probe_th06c.py tests/test_touhou_probe_cli.py
git commit -m "feat: add th06c known-RVA sanity probe"
```

---

### Task 5: Typed Candidate Scanner for Unknown Values

**Files:**
- Create: `src/flysurvivors/touhou/scanner.py`
- Test: `tests/test_touhou_scanner.py`

**Interfaces:**
- Produces `ValueType`, `ScanMode`, `CandidateSet`, `initial_scan(...)`, `rescan(...)`, `save_candidates(...)`, `load_candidates(...)`.
- Scanner consumes any `ProcessMemory`, not specifically Windows.
- Candidate persistence stores addresses, last observed values, scalar type, and process-module identity in compressed `.npz` files.

- [ ] **Step 1: Write failing scanner tests with synthetic memory**

```python
# tests/test_touhou_scanner.py
import struct

import numpy as np

from flysurvivors.touhou.memory import MemoryRegion
from flysurvivors.touhou.scanner import initial_scan, load_candidates, rescan, save_candidates


class FakeMemory:
    pid = 77
    module_base = 0x1000

    def __init__(self, data: bytearray):
        self.data = data

    def regions(self):
        yield MemoryRegion(0x1000, len(self.data), True, True, False)

    def read(self, address, size):
        off = address - 0x1000
        return bytes(self.data[off:off + size])

    def close(self):
        pass


def test_i32_exact_scan_then_rescan_changed(tmp_path):
    raw = bytearray(64)
    struct.pack_into("<i", raw, 8, 5)
    struct.pack_into("<i", raw, 20, 5)
    mem = FakeMemory(raw)
    c = initial_scan(mem, value_type="i32", mode="eq", target=5)
    assert set(c.addresses.tolist()) == {0x1008, 0x1014}

    struct.pack_into("<i", raw, 8, 4)
    c2 = rescan(mem, c, mode="changed")
    assert c2.addresses.tolist() == [0x1008]

    path = tmp_path / "lives.npz"
    save_candidates(path, c2)
    loaded = load_candidates(path)
    assert np.array_equal(loaded.addresses, c2.addresses)


def test_f32_scan_can_filter_finite_range():
    raw = bytearray(64)
    struct.pack_into("<f", raw, 4, -12.5)
    struct.pack_into("<f", raw, 12, 9000.0)
    mem = FakeMemory(raw)
    c = initial_scan(mem, value_type="f32", mode="range", minimum=-100.0, maximum=100.0)
    assert 0x1004 in c.addresses
    assert 0x100C not in c.addresses
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
python -m pytest tests/test_touhou_scanner.py -v
```

Expected: import failure for `scanner.py`.

- [ ] **Step 3: Implement vectorized aligned scanning**

Define:

```python
ValueType = Literal["u8", "i32", "u32", "f32", "f64"]
ScanMode = Literal["eq", "range", "changed", "unchanged", "increased", "decreased"]

@dataclass
class CandidateSet:
    addresses: np.ndarray
    values: np.ndarray
    value_type: ValueType
    pid: int
    module_base: int
```

Implementation rules:

- Scan only `MemoryRegion.readable` regions.
- Default to writable regions for gameplay-state discovery; expose `writable_only: bool = True`.
- Use natural alignment (`1`, `4`, `4`, `4`, `8` bytes respectively).
- Convert chunks with `np.frombuffer` using explicit little-endian dtypes.
- For floating types, discard non-finite values before comparisons.
- `changed`/`unchanged` compare against `CandidateSet.values`; `increased`/`decreased` use numeric comparisons.
- Rescan only existing candidate addresses; do not rescan the entire process.
- Failed reads during rescan drop that candidate instead of fabricating a value.
- `.npz` persistence must include a small integer format version and reject unsupported versions.

Use 1 MiB chunks so a large committed region is never copied as one giant Python allocation.

- [ ] **Step 4: Run scanner tests**

```powershell
python -m pytest tests/test_touhou_scanner.py -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite to catch package regressions**

```powershell
python -m pytest -q
```

Expected: all existing and new tests pass; GPU-only tests may retain their existing skip behavior where applicable.

- [ ] **Step 6: Commit**

```powershell
git add src/flysurvivors/touhou/scanner.py tests/test_touhou_scanner.py
git commit -m "feat: add typed th06c memory candidate scanner"
```

---

### Task 6: Scanner CLI Commands and Human-Guided Discovery Workflow

**Files:**
- Modify: `src/flysurvivors/touhou/probe_cli.py`
- Modify: `tests/test_touhou_probe_cli.py`
- Modify: `scripts/probe_th06c.py`

**Interfaces:**
- Adds CLI commands `scan`, `rescan`, and `watch`.
- `scan` always verifies the executable hash before attaching.
- `rescan` rejects candidate files whose saved module identity does not match the currently attached process.
- `watch` observes explicit addresses/RVAs only; it never writes memory.

- [ ] **Step 1: Add failing CLI parser tests**

```python
# append to tests/test_touhou_probe_cli.py

def test_parser_accepts_exact_i32_scan():
    args = build_parser().parse_args([
        "scan", "--exe", "C:/x/th06c.exe", "--type", "i32",
        "--eq", "5", "--out", "lives.npz",
    ])
    assert args.command == "scan"
    assert args.value_type == "i32"
    assert args.eq == "5"


def test_parser_accepts_changed_rescan():
    args = build_parser().parse_args([
        "rescan", "--exe", "C:/x/th06c.exe", "--in", "lives.npz",
        "--changed", "--out", "lives2.npz",
    ])
    assert args.command == "rescan"
    assert args.changed
```

- [ ] **Step 2: Run tests and verify parser failure**

```powershell
python -m pytest tests/test_touhou_probe_cli.py -v
```

Expected: new parser assertions fail.

- [ ] **Step 3: Implement explicit scanning commands**

Required command shapes:

```text
probe_th06c.py scan   --exe PATH --type i32 --eq 5 --out lives_5.npz
probe_th06c.py scan   --exe PATH --type f32 --min -500 --max 500 --out player_xy_0.npz
probe_th06c.py rescan --exe PATH --in lives_5.npz --eq 4 --out lives_4.npz
probe_th06c.py rescan --exe PATH --in player_xy_0.npz --changed --out player_xy_1.npz
probe_th06c.py rescan --exe PATH --in player_xy_1.npz --unchanged --out player_xy_2.npz
probe_th06c.py watch  --exe PATH --rva 0x003A3B4C --type u32 --interval 0.25
```

Rules:

- Exactly one scan predicate is allowed per invocation.
- Parse integer literals with `int(text, 0)` so decimal and `0x...` work.
- Parse floating targets with `float(text)`.
- Print candidate counts before and after each scan.
- `watch` prints timestamp, address, and current value until Ctrl+C; Ctrl+C exits cleanly with code 0.
- Never default to any unknown gameplay RVA.

- [ ] **Step 4: Run focused CLI/scanner tests**

```powershell
python -m pytest tests/test_touhou_probe_cli.py tests/test_touhou_scanner.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/flysurvivors/touhou/probe_cli.py scripts/probe_th06c.py tests/test_touhou_probe_cli.py
git commit -m "feat: expose th06c memory discovery CLI"
```

---

### Task 7: Document and Execute the First Real Discovery Protocol

**Files:**
- Create: `docs/th06c-probe.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces a reproducible human procedure for validating the probe and beginning the player/lives search.
- Does not claim any newly discovered RVA until it has been behaviorally verified.

- [ ] **Step 1: Add local probe-output exclusions**

Append to `.gitignore`:

```gitignore
# Local Touhou memory-probe artifacts
probe-data/
*.probe.npz
```

- [ ] **Step 2: Write the exact PowerShell workflow**

`docs/th06c-probe.md` must include these commands verbatim with explanation:

```powershell
$Exe = "C:\Program Files (x86)\Steam\steamapps\common\th06c\th06c.exe"
python scripts/probe_th06c.py info --exe $Exe
python scripts/probe_th06c.py known --exe $Exe
```

For remaining lives, use the configured starting value of 5 as a candidate-discovery experiment:

```powershell
New-Item -ItemType Directory -Force probe-data | Out-Null
python scripts/probe_th06c.py scan --exe $Exe --type i32 --eq 5 --out probe-data\lives_5.probe.npz
# Lose exactly one life in-game, then immediately run:
python scripts/probe_th06c.py rescan --exe $Exe --in probe-data\lives_5.probe.npz --eq 4 --out probe-data\lives_4.probe.npz
# Change menus/play state without losing another life; values representing lives should remain 4:
python scripts/probe_th06c.py rescan --exe $Exe --in probe-data\lives_4.probe.npz --unchanged --out probe-data\lives_stable.probe.npz
```

For player-position discovery, document a separate float experiment rather than mixing it with lives:

```powershell
python scripts/probe_th06c.py scan --exe $Exe --type f32 --min -1000 --max 1000 --out probe-data\player_0.probe.npz
# Move horizontally while keeping the vertical position as stable as practical:
python scripts/probe_th06c.py rescan --exe $Exe --in probe-data\player_0.probe.npz --changed --out probe-data\player_1.probe.npz
# Stop moving and pause/hold position, then keep only stable candidates:
python scripts/probe_th06c.py rescan --exe $Exe --in probe-data\player_1.probe.npz --unchanged --out probe-data\player_2.probe.npz
```

The document must state that these scans produce **candidates**, not accepted offsets. Acceptance requires repeated controlled movement/death observations and explicit watching of a candidate while comparing it with the game.

- [ ] **Step 3: Add an acceptance checklist to the document**

The checklist must require all of the following before recording an RVA as verified:

```text
[ ] Candidate is inside the verified th06c process and current module/session context.
[ ] Value changes exactly when the corresponding visible game quantity changes.
[ ] Value remains stable during unrelated actions.
[ ] Behavior repeats across at least three fresh game runs.
[ ] Address can be expressed reproducibly as module_base + RVA, or a documented pointer chain/signature if not module-relative.
[ ] No write/patch/input operation was used to make the candidate fit expectations.
```

- [ ] **Step 4: Run the full automated suite**

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Run the real read-only smoke workflow on Windows**

With the exact verified Steam build running, execute `info`, `known`, and the lives/player candidate workflows above. Record only observations in local notes; do not add guessed offsets to source code.

Expected deliverable: the probe remains stable while Touhou runs, known-RVA values are readable, candidate sets narrow across controlled observations, and the game is never modified.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore docs/th06c-probe.md
git commit -m "docs: add th06c probe workflow"
```

---

## Plan Self-Review

### Spec coverage for this subproject

This plan implements only Phase A probe infrastructure from the approved architecture: exact-build guard, ASLR-safe RVA resolution, read-only attachment, known-RVA sanity checks, unknown-value discovery tooling, and a reproducible validation procedure. It intentionally does **not** implement the independent subsystems for DLL injection, object-pool extraction, retina generation, LAN transport, motor control, lifecycle automation, persistent history, or 3D visualization. Those receive separate implementation plans after the probe has produced verified object layouts/addresses.

### Placeholder scan

No `TBD`, `TODO`, guessed gameplay offsets, or unspecified implementation steps are permitted in this plan. Unknown game addresses remain runtime discovery results rather than placeholders in source.

### Type consistency

- `WindowsProcessMemory` implements the `ProcessMemory` protocol.
- `scanner.py` consumes `ProcessMemory` and returns `CandidateSet`.
- CLI handlers compose `verify_build` -> `WindowsProcessMemory.attach` -> typed readers/scanner operations.
- All persisted candidates carry `pid`, `module_base`, and scalar type so a rescan can reject incompatible session context rather than silently reading unrelated addresses.
