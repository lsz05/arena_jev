"""The arena website: /arenas/ (index), /arenas/uno/ (UNO viewer), /arenas/leaderboard/, /arenas/bench/.

    python -m arena site --uno-run runs/<dir> [--base /arenas] [--port 8200] [--allowed-host jev.takeshiba.dev]

--uno-run may be "latest" (the newest runs/*_swiss directory, re-resolved when a newer one appears).
Pages share arena/static/ (served at <base>/static/). The leaderboard page reads the UNO run's leaderboard.json
(or the registry and JevBench results alone before any tournament); the bench page reads runs/jevbench-*.
"""

from __future__ import annotations

import json
import mimetypes
from http.server import ThreadingHTTPServer
from pathlib import Path

from . import benchdata, leaderboard
from .viz import Response, UnoApp, as_json, make_handler

STATIC = Path(__file__).with_name("static")
RUNS = Path(__file__).resolve().parent.parent / "runs"


class Site:
    def __init__(self, uno_run: str | None, base: str = "/arenas"):
        self.base = "/" + base.strip("/")
        self.uno_run = uno_run
        self.uno: UnoApp | None = None
        self.uno_dir: Path | None = None

    def _uno(self) -> UnoApp | None:
        target = self.uno_run
        if target == "latest":
            runs = sorted(RUNS.glob("*_swiss"))
            target = str(runs[-1]) if runs else None
        if target and (self.uno_dir is None or Path(target) != self.uno_dir):
            if (Path(target) / "games.jsonl").exists():
                self.uno, self.uno_dir = UnoApp(target), Path(target)
        return self.uno

    def board(self) -> dict:
        uno = self._uno()
        return uno.board() if uno else leaderboard.build(None)

    def route(self, path: str) -> Response:
        b = self.base
        if path in (b, f"{b}/uno", f"{b}/leaderboard", f"{b}/bench"):
            return 301, "", (path + "/").encode()
        if not path.startswith(b + "/"):
            return 404, "text/plain", b"not found"
        path = path[len(b):]
        if path == "/":
            return self.static("index.html")
        if path.startswith("/static/"):
            return self.static(path[len("/static/"):])
        if path.startswith("/uno/"):
            uno = self._uno()
            if uno is None:
                return 503, "text/plain", b"no UNO tournament yet"
            return uno.get(path[len("/uno"):])
        if path == "/leaderboard/":
            return self.static("leaderboard.html")
        if path == "/leaderboard/api/leaderboard":
            return as_json(self.board())
        if path == "/bench/":
            return self.static("bench.html")
        if path == "/bench/api/bench":
            return as_json(benchdata.build())
        if path == "/api/summary":  # for the index page
            board = self.board()
            bench = benchdata.build()
            return as_json({"models": len(board["rows"]), "games": board["games"], "updated": board["updated"],
                            "in_progress": board["in_progress"], "target": board.get("target_games_per_model"),
                            "bench_models": len(bench["models"])})
        return 404, "text/plain", b"not found"

    @staticmethod
    def static(name: str) -> Response:
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            return 404, "text/plain", b"not found"
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        return 200, ctype, path.read_bytes()


def serve(uno_run: str | None, base: str = "/arenas", host: str = "127.0.0.1", port: int = 8200,
          allowed_hosts: list[str] | None = None) -> None:
    site = Site(uno_run, base)
    print(f"arena site on http://{host}:{port}{site.base}/ (UNO run: {uno_run})", flush=True)
    ThreadingHTTPServer((host, port), make_handler(site.route, allowed_hosts)).serve_forever()
