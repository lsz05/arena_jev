"""Game viewer: results, game list, and step-by-step replays with each model's view of the decision.

    python -m arena viz runs/<dir>                       serve on http://127.0.0.1:8200
    python -m arena viz runs/<dir> --base-path /arenas/uno   also serve under a path prefix (reverse proxies
                                                         that pass the full path, e.g. a Cloudflare Tunnel path rule)
    python -m arena viz runs/<dir> --export out.html     one self-contained file with a sample of games

The whole site (UNO viewer, leaderboard, bench) is served by `python -m arena site` (arena/site.py).
"""

from __future__ import annotations

import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from . import leaderboard
from .replay import RunData

PAGE = Path(__file__).with_name("viewer.html")

Response = tuple[int, str, bytes]


def run_payload(data: RunData) -> dict:
    return {
        "name": data.run.get("name", data.dir.name),
        "dir": data.dir.name,
        "run": data.run,
        "players": list(data.specs.values()),
        "summary": data.summary(),
        "servers": data.servers(),
    }


def as_json(obj) -> Response:
    return 200, "application/json", json.dumps(obj, ensure_ascii=False).encode()


class UnoApp:
    """The UNO viewer for one run directory: the page plus its /api/* endpoints (paths relative to the app)."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.data = RunData(run_dir)
        self.replay = lru_cache(maxsize=64)(self.data.replay)
        self.board_path = self.run_dir / "leaderboard.json"
        self.built: dict = {}

    def board(self) -> dict:
        # leaderboard.json is rewritten while a run is in progress, so read it fresh on every request
        if self.board_path.exists():
            return json.loads(self.board_path.read_text())
        if not self.built:
            self.built.update(leaderboard.build(self.run_dir))
        return self.built

    def get(self, path: str) -> Response:
        data = self.data
        if path in ("/", "/index.html"):
            return 200, "text/html; charset=utf-8", PAGE.read_bytes()
        if path == "/api/run":
            data.refresh()
            return as_json(run_payload(data))
        if path == "/api/games":
            data.refresh()
            return as_json(data.index())
        if path == "/api/leaderboard":
            return as_json(self.board())
        if path.startswith("/api/game/"):
            gid = path[len("/api/game/"):]
            if gid not in data.games:
                data.refresh()
            if gid not in data.games:
                return 404, "text/plain", b"unknown game"
            return as_json(self.replay(gid))
        return 404, "text/plain", b"not found"


def make_handler(route, allowed_hosts: list[str] | None):
    """A request handler around `route(path) -> Response`. With `allowed_hosts`, requests for any other Host
    (local requests excepted) get a 404, so a stale reverse-proxy route cannot reach the site."""
    allowed = {h.lower() for h in allowed_hosts or []} | {"127.0.0.1", "localhost"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if allowed_hosts and self.headers.get("Host", "").rsplit(":", 1)[0].lower() not in allowed:
                return self._send(404, "text/plain", b"not found")
            path = unquote(self.path.split("?", 1)[0])
            try:
                code, ctype, body = route(path)
            except Exception as e:  # show the error in the page rather than a dropped connection
                code, ctype, body = 500, "text/plain", f"{type(e).__name__}: {e}".encode()
            if code in (301, 302):
                self.send_response(code)
                self.send_header("Location", body.decode())
                self.end_headers()
                return
            self._send(code, ctype, body)

        def _send(self, code: int, ctype: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")  # revalidate: a proxy (Cloudflare) must not serve stale pages
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def serve(run_dir: str, host: str = "127.0.0.1", port: int = 8200, base_path: str = "",
          allowed_hosts: list[str] | None = None) -> None:
    base = "/" + base_path.strip("/") if base_path.strip("/") else ""
    app = UnoApp(run_dir)

    def route(path: str) -> Response:
        if base and path == base:  # relative API URLs need the trailing slash
            return 301, "", (base + "/").encode()
        if base and path.startswith(base + "/"):
            path = path[len(base):]
        return app.get(path)

    print(f"viewer for {app.run_dir} ({len(app.data.games)} games): http://{host}:{port}{base}/", flush=True)
    ThreadingHTTPServer((host, port), make_handler(route, allowed_hosts)).serve_forever()


def export(run_dir: str, out: str, deals: int = 3) -> Path:
    """Write one HTML file with the run's results, the full game list, and replays of the first
    `deals` deals (every seat rotation of each, so the same cards can be compared across seats)."""
    data = RunData(run_dir)
    chosen = [gid for gid in data.games if _deal(gid) < deals]
    board_path = Path(run_dir) / "leaderboard.json"
    embed = {
        "run": run_payload(data),
        "leaderboard": json.loads(board_path.read_text()) if board_path.exists() else leaderboard.build(run_dir),
        "games": data.index(),
        "replays": {gid: data.replay(gid) for gid in chosen},
    }
    blob = json.dumps(embed, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.read_text().replace("<!--EMBED-->", f"<script>window.ARENA_EMBED = {blob};</script>")
    path = Path(out)
    path.write_text(page)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(chosen)} replays)")
    return path


def _deal(game_id: str) -> int:
    """Deal (table-format runs, "L0-D12-R3") or table (Swiss runs, "T00012-R3") number of a game id."""
    part = next((p for p in game_id.split("-") if p[:1] in ("D", "T") and p[1:].isdigit()), "D0")
    return int(part[1:])
