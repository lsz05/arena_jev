"""Drop this user's model files from the page cache (no root needed), so a starting server can get GPU memory.

    python3 servers/dropcache.py [dir ...]     (default: the Hugging Face hub cache, servers/src and servers/venvs)

On the GB10 CPU and GPU share one memory pool, and cudaMalloc does not reclaim the page cache: with the cache full a
new server fails to load ("cudaErrorMemoryAllocation") although `free` shows memory as available. posix_fadvise
DONTNEED on files we can read evicts their clean cached pages; running processes keep their loaded weights.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT = [Path.home() / ".cache" / "huggingface" / "hub", ROOT / "src", ROOT / "venvs"]


def drop(dirs) -> int:
    n = 0
    for d in dirs:
        for dirpath, _, files in os.walk(d):
            for f in files:
                path = os.path.join(dirpath, f)
                try:
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                except OSError:
                    continue
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                except OSError:
                    pass
                finally:
                    os.close(fd)
    return n


if __name__ == "__main__":
    print(f"dropped the cached pages of {drop([Path(a) for a in sys.argv[1:]] or DEFAULT)} files")
