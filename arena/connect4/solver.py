"""Python access to the C Connect Four solver (solver.c), compiled on first use.

    solve(moves, weak=True)    -> score of the position for the side to move (sign: win / draw / loss)
    analyze(moves, weak=True)  -> [score of playing column c for c in 0..6], None for a full column

Moves are 0-based columns. With weak=True only the sign of a score is exact, which is enough to tell winning, drawing
and losing moves apart. The solver keeps one table per process and is not thread-safe: calls are serialized with a
lock; use processes for parallel work.
"""

from __future__ import annotations

import ctypes
import subprocess
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "solver.c"
LIBRARY = HERE / "_c4solver.so"
UNPLAYABLE, INVALID = -1000, -10000
_lib = None
_lock = threading.Lock()


def _load():
    global _lib
    if _lib is None:
        if not LIBRARY.exists() or LIBRARY.stat().st_mtime < SOURCE.stat().st_mtime:
            tmp = LIBRARY.with_suffix(".tmp.so")
            subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", str(tmp), str(SOURCE)], check=True)
            tmp.replace(LIBRARY)
        lib = ctypes.CDLL(str(LIBRARY))
        lib.c4_solve.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.c4_solve.restype = ctypes.c_int
        lib.c4_analyze.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        lib.c4_analyze.restype = ctypes.c_int
        _lib = lib
    return _lib


def _encode(moves) -> bytes:
    return "".join(str(c + 1) for c in moves).encode()


def solve(moves, weak: bool = True) -> int:
    with _lock:
        score = _load().c4_solve(_encode(moves), int(weak))
    if score == INVALID:
        raise ValueError(f"not a position in play: {list(moves)}")
    return score


def analyze(moves, weak: bool = True) -> list[int | None]:
    out = (ctypes.c_int * 7)()
    with _lock:
        ok = _load().c4_analyze(_encode(moves), int(weak), out)
    if ok != 0:
        raise ValueError(f"not a position in play: {list(moves)}")
    return [None if v == UNPLAYABLE else v for v in out]
