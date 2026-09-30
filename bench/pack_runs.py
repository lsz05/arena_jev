"""Compress run records for the git repository, and restore them.

    python bench/pack_runs.py pack   runs/<run> [...]   games.jsonl / decisions.jsonl / <model>/results.jsonl
                                                         -> *.jsonl.xz (split above 45 MB)
    python bench/pack_runs.py unpack runs/<run> [...]   the reverse, so the viewer can read the run again

GitHub warns about files above 50 MB and refuses them above 100 MB, so a large archive is split into
<name>.jsonl.xz.00, .01, ... (plain byte chunks: `cat` them back together, or use `unpack`).
"""

import lzma
import sys
from pathlib import Path

CHUNK = 45 * 1024 * 1024
NAMES = ("games.jsonl", "decisions.jsonl", "*/results.jsonl")  # the last: one file per model (puzzle runs)


def pack(run: Path) -> None:
    for src in [p for name in NAMES for p in sorted(run.glob(name))]:
        for old in src.parent.glob(src.name + ".xz*"):
            old.unlink()
        data = lzma.compress(src.read_bytes(), preset=9 | lzma.PRESET_EXTREME)
        if len(data) <= CHUNK:
            (src.parent / (src.name + ".xz")).write_bytes(data)
            parts = 1
        else:
            parts = 0
            for i in range(0, len(data), CHUNK):
                (src.parent / f"{src.name}.xz.{parts:02d}").write_bytes(data[i:i + CHUNK])
                parts += 1
        print(f"{src}: {src.stat().st_size / 1e6:.1f} MB -> {len(data) / 1e6:.1f} MB xz in {parts} file(s)")


def unpack(run: Path) -> None:
    targets = {p.parent / p.name.split(".xz")[0] for name in NAMES for p in run.glob(name + ".xz*")}
    for dst in sorted(targets):
        parts = sorted(dst.parent.glob(dst.name + ".xz.[0-9][0-9]")) or sorted(dst.parent.glob(dst.name + ".xz"))
        data = lzma.decompress(b"".join(p.read_bytes() for p in parts))
        dst.write_bytes(data)
        print(f"{dst}: {len(data) / 1e6:.1f} MB from {len(parts)} file(s)")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("pack", "unpack"):
        sys.exit(__doc__)
    for arg in sys.argv[2:]:
        (pack if sys.argv[1] == "pack" else unpack)(Path(arg))
