"""Swiss-style UNO tournament over many models, scheduled asynchronously.

Every model sits at one table at a time. Whenever at least `num_players` models are free, a table opens:
the free model with the fewest games anchors it, and the other seats go to the free models whose current
rating is closest to the anchor's (TrueSkill mu - 3 sigma plus Gaussian jitter, so pairings vary). Models that
already reached the target only fill seats nobody else can take. A table plays one deal once per seat rotation
(4 games for 4 players, run in parallel). There are no global rounds, so a slow model never makes the others wait.

After each table: TrueSkill (winner first, the others by points left in hand), running statistics and
leaderboard.json / summary.json / status.json are updated, so the viewer can show the tournament live.

A model whose decisions fail (fallback) `restart_after` times in a row is restarted once through
servers/launch.py; if it keeps failing it is retired from scheduling and marked in status.json.

    python -m arena swiss configs/swiss.toml
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


from . import leaderboard, stats, tokens
from .match import play_game
from .players import SystemOnePlayer
from .systemone import SystemOneClient
from .uno.render import TEMPLATE_VERSION

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "servers" / "models"


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    run = cfg.setdefault("run", {})
    run.setdefault("name", Path(path).stem)
    run.setdefault("num_players", 4)
    run.setdefault("target_games", 2000)
    run.setdefault("seed", 0)
    run.setdefault("template", "B")
    run.setdefault("history_rounds", 5)
    run.setdefault("policy", "argmax")
    run.setdefault("wd4_challenge", True)
    run.setdefault("shuffle_hand", True)
    run.setdefault("jitter", 2.0)
    run.setdefault("restart_after", 20)
    run.setdefault("max_lead", 40)
    run.setdefault("max_steps", 1000)
    run.setdefault("must_play", False)  # house rule: a playable card must be played (no drawing instead)  # a game still running after this many decisions is aborted (not rated)  # games a model may be ahead of the one furthest behind (0: no limit)
    run.setdefault("budget_tokenizer", tokens.DEFAULT_TOKENIZER)
    run.setdefault("models", [])  # empty: every servers/models/*.toml
    run.setdefault("exclude", [])
    return cfg


def model_specs(run: dict) -> list[dict]:
    specs = {}
    for f in sorted(MODELS_DIR.glob("*.toml")):
        with open(f, "rb") as fh:
            spec = tomllib.load(fh)
        specs[spec["name"]] = spec
    names = run["models"] or list(specs)
    return [specs[n] for n in names if n in specs and n not in run["exclude"]]


def make_player(spec: dict, reg: dict, run: dict) -> SystemOnePlayer:
    limit = spec.get("max_input_tokens") or reg.get("max_input_tokens")
    tok = spec.get("tokenizer") or reg.get("tokenizer") or run["budget_tokenizer"]
    client = SystemOneClient(f"http://127.0.0.1:{spec['port']}", model=spec["name"], timeout=120, retries=2)
    opt_limit = spec.get("max_option_tokens") or reg.get("max_option_tokens")
    return SystemOnePlayer(spec["name"], client, template=run["template"], history_rounds=run["history_rounds"],
                           policy=run["policy"], max_options=spec.get("max_options") or reg.get("max_options") or 255,
                           max_prompt_tokens=limit, count_tokens=tokens.counter(tok) if limit or opt_limit else None,
                           max_option_tokens=opt_limit)


class Tournament:
    def __init__(self, cfg: dict, out: Path, players: dict[str, SystemOnePlayer]):
        self.cfg, self.run, self.out = cfg, cfg["run"], out
        self.players = players
        self.env = leaderboard.rating_env()  # tau = 0: a model's skill does not drift
        self.ratings = {n: self.env.create_rating() for n in players}
        self.opponents: dict[str, list[str]] = defaultdict(list)
        self.games_played = Counter()
        self.busy: set[str] = set()
        self.retired: dict[str, str] = {}
        self.fail_streak = Counter()
        self.restarted: set[str] = set()
        self.agg = stats.Aggregator()
        self.table_no = 0
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.rng = random.Random(self.run["seed"])
        self.games_f = open(out / "games.jsonl", "a")
        self.decisions_f = open(out / "decisions.jsonl", "a")
        self.registry = leaderboard.load_registry()
        self.started = time.time()
        self._hold: tuple[float, set[str]] = (0.0, set())

    def restore(self) -> None:
        """Continue a stopped run: replay its finished games (ratings, counts, stats). Decisions of tables that never
        finished stay in decisions.jsonl but are not counted, and new tables are numbered after every table seen."""
        games = [json.loads(line) for line in open(self.out / "games.jsonl") if line.strip()]
        done = {g["game"] for g in games}
        top = max((g["table"] for g in games), default=0)
        for line in open(self.out / "decisions.jsonl"):
            if not line.endswith("\n"):
                break
            d = json.loads(line)
            top = max(top, int(d["game"][1:].split("-")[0]))
            if d["game"] in done:
                self.agg.add_decision(d)
        for g in games:
            for m in g["seats"]:
                if m not in self.ratings:  # a model that is not back this time still keeps its games
                    self.ratings[m] = self.env.create_rating()
            self._apply(g)
        self.table_no = top
        print(f"resumed {len(games)} games, next table {top + 1}", flush=True)

    def held(self) -> set[str]:
        """Models listed in <run>/hold.txt get no new table (to restart a server safely: hold, wait, restart, release)."""
        path = self.out / "hold.txt"
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return set()
        if mtime != self._hold[0]:
            self._hold = (mtime, {x.strip() for x in path.read_text().split() if x.strip()})
        return self._hold[1]

    # ---- scheduling -----------------------------------------------------------------------------

    def score(self, name: str) -> float:
        r = self.ratings[name]
        return r.mu - 3 * r.sigma + self.rng.gauss(0, self.run["jitter"])

    def pick_table(self) -> list[str] | None:
        n, target = self.run["num_players"], self.run["target_games"]
        held = self.held()
        free = [m for m in self.players if m not in self.busy and m not in self.retired and m not in held]
        needing = [m for m in free if self.games_played[m] < target]
        if len(free) < n or not needing:
            return None
        fewest = min(self.games_played[m] for m in needing)
        # keep game counts even: no new table while its anchor is max_lead games ahead of the model furthest behind
        # (slow servers would otherwise fall far behind; the GPU is shared, so waiting fast models speed the slow ones up)
        behind = [self.games_played[m] for m in self.players if m not in self.retired and m not in held]
        if self.run["max_lead"] and behind and fewest >= min(behind) + self.run["max_lead"]:
            return None
        anchor = self.rng.choice([m for m in needing if self.games_played[m] == fewest])
        a = self.score(anchor)
        others = [m for m in free if m != anchor]
        # candidates still below target first, then by closeness of (noisy) rating to the anchor
        others.sort(key=lambda m: (self.games_played[m] >= target, abs(self.score(m) - a)))
        return [anchor] + others[: n - 1]

    def discover(self) -> None:
        """Add model servers that appeared since the start (runs in a background thread; pings can be slow)."""
        registry = leaderboard.load_registry()
        for spec in model_specs(self.run):
            if spec["name"] in self.players or spec["name"] in self.retired:
                continue
            if not _answers(spec):
                continue
            player = make_player(spec, registry.get(spec["name"], {}), self.run)
            with self.cond:
                self.registry = registry
                _add_players(self.out, {spec["name"]: player})
                self.players[spec["name"]] = player
                self.ratings[spec["name"]] = self.env.create_rating()
                print(f"{spec['name']} joined the tournament", flush=True)
                self.cond.notify_all()

    def done(self) -> bool:
        active = [m for m in self.players if m not in self.retired]
        return all(self.games_played[m] >= self.run["target_games"] for m in active)

    # ---- one table ------------------------------------------------------------------------------

    def play_table(self, seats: list[str], table_no: int) -> None:
        n = len(seats)
        seed = self.run["seed"] * 1_000_003 + table_no
        order = seats[:]
        self.rng.shuffle(order)
        lineups = [order[r:] + order[:r] for r in range(n)]
        results: list[dict] = []

        def one(r: int) -> dict:
            return play_game([self.players[m] for m in lineups[r]], seed, game_id=f"T{table_no:05d}-R{r}",
                             meta={"table": table_no, "rotation": r}, on_decision=self.on_decision,
                             wd4_challenge=self.run["wd4_challenge"], shuffle_hand=self.run["shuffle_hand"],
                             max_steps=self.run["max_steps"], must_play=self.run["must_play"])

        try:
            with ThreadPoolExecutor(max_workers=n) as pool:
                results = list(pool.map(one, range(n)))
        except Exception as e:  # noqa: BLE001 - a crashed table must not stop the tournament
            print(f"table {table_no} {seats} failed: {type(e).__name__}: {e}", flush=True)
        with self.cond:
            for g in results:
                self.record_game(g)
            self.busy.difference_update(seats)
            self.write_status()
            self.cond.notify_all()

    def on_decision(self, d: dict) -> None:
        line = json.dumps(d, ensure_ascii=False)
        with self.lock:
            self.decisions_f.write(line + "\n")
            self.agg.add_decision(d)
            if d.get("calls") is None and "fallback" not in d:
                return
            name = d["player"]
            if d.get("fallback") and _contract_error(d.get("error")):
                return  # e.g. 422 for an over-long input: restarting the server would not change it
            self.fail_streak[name] = self.fail_streak[name] + 1 if d.get("fallback") else 0
            if self.fail_streak[name] >= self.run["restart_after"] and name not in self.retired:
                self.fail_streak[name] = 0
                threading.Thread(target=self.recover, args=(name, d.get("error")), daemon=True).start()

    def recover(self, name: str, error: str | None) -> None:
        if name in self.restarted:
            with self.lock:
                self.retired[name] = f"retired after repeated failures: {error}"
            print(f"retiring {name}: {error}", flush=True)
            return
        self.restarted.add(name)
        print(f"restarting {name} after {self.run['restart_after']} failed decisions: {error}", flush=True)
        launch = [sys.executable, str(ROOT / "servers" / "launch.py")]
        spec = next((s for s in model_specs(self.run) if s["name"] == name), None)
        for attempt in range(3):  # a start can fail at random (e.g. a Triton kernel build racing with another process)
            subprocess.run(launch + ["stop", name], capture_output=True)
            time.sleep(3)
            # free the page cache first: on the GB10 a full cache makes the new server's cudaMalloc fail
            subprocess.run([sys.executable, str(ROOT / "servers" / "dropcache.py")], capture_output=True)
            subprocess.run(launch + ["start", name], capture_output=True)
            subprocess.run(launch + ["wait", name], capture_output=True)
            if spec is None or _answers(spec):
                print(f"{name} is back (attempt {attempt + 1})", flush=True)
                return
        print(f"{name} did not come back after 3 attempts", flush=True)

    def record_game(self, g: dict) -> None:
        self.games_f.write(json.dumps(g, ensure_ascii=False) + "\n")
        self._apply(g)

    def _apply(self, g: dict) -> None:
        self.agg.add_game(g)
        seats = g["seats"]
        for m in seats:
            self.games_played[m] += 1
            self.opponents[m] += [o for o in seats if o != m]
        if g["aborted"]:
            return
        pts = g["hand_points"]
        keys = [0 if s == g["winner_seat"] else 1 + pts[s] for s in range(len(seats))]
        order = sorted(set(keys))
        new = self.env.rate([(self.ratings[m],) for m in seats], ranks=[order.index(k) for k in keys])
        for m, (r,) in zip(seats, new):
            self.ratings[m] = r

    # ---- files ----------------------------------------------------------------------------------

    def write_status(self, state: str = "running") -> None:
        self.games_f.flush()
        self.decisions_f.flush()
        meta = {
            "state": state,
            "target_games_per_model": self.run["target_games"],
            "started": datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(time.time() - self.started),
            "games": self.agg.games,
            "tables": self.table_no,
            "busy": sorted(self.busy),
            "retired": self.retired,
            "games_per_model": dict(self.games_played),
        }
        _atomic(self.out / "status.json", meta)
        summary = self.agg.summary()
        _atomic(self.out / "summary.json", summary)
        board = leaderboard.assemble(summary, self.ratings, self.opponents, self.registry, self.out, self.agg.games,
                                     0, meta)
        _atomic(self.out / "leaderboard.json", board)

    # ---- main loop ------------------------------------------------------------------------------

    def run_all(self) -> None:
        with ThreadPoolExecutor(max_workers=16) as pool, self.cond:  # tables in flight are limited by free models
            self.write_status()
            last_print = last_scan = time.time()
            scan = None
            while not self.done():
                if time.time() - last_scan > 300 and not (scan and scan.is_alive()):
                    last_scan = time.time()
                    scan = threading.Thread(target=self.discover, daemon=True)
                    scan.start()
                seats = self.pick_table()
                if seats is None:
                    if not self.busy and not self.done() and not self.held():
                        print("no table can be formed and nothing is running; stopping", flush=True)
                        break
                    self.cond.wait(timeout=30)
                else:
                    self.table_no += 1
                    self.busy.update(seats)
                    pool.submit(self.play_table, seats, self.table_no)
                if time.time() - last_print > 300:
                    last_print = time.time()
                    lo = min((self.games_played[m] for m in self.players if m not in self.retired), default=0)
                    print(f"[{datetime.now():%H:%M:%S}] {self.agg.games} games, {self.table_no} tables, "
                          f"fewest games per model {lo}/{self.run['target_games']}", flush=True)
            while self.busy:
                self.cond.wait(timeout=30)
            self.write_status("done")
        self.games_f.close()
        self.decisions_f.close()


def _atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _add_players(out: Path, players: dict[str, SystemOnePlayer]) -> None:
    """Record the players in config.toml and servers.json, so the viewer can rebuild their prompts."""
    with open(out / "config.toml", "a") as f:
        for name, p in players.items():
            f.write(f'\n[[players]]\nname = "{name}"\ntype = "systemone"\nbase_url = "{p.client.base_url}"\n')
    info = json.loads((out / "servers.json").read_text())
    for name, p in players.items():
        try:
            models = p.client.models()
        except Exception:  # noqa: BLE001
            models = None
        info["servers"][name] = {"base_url": p.client.base_url, "models": models}
    _atomic(out / "servers.json", info)


def _contract_error(error: str | None) -> bool:
    m = re.search(r"HTTP (4\d\d)", error or "")
    return bool(m) and m.group(1) != "429"


def _answers(spec: dict) -> bool:
    client = SystemOneClient(f"http://127.0.0.1:{spec['port']}", model=spec["name"], timeout=60, retries=0)
    try:
        client.evaluate("ping", {"q": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "A", "b": "B"}}})
        return True
    except Exception:  # noqa: BLE001
        return False


def run(config_path: str | Path, out_root: str | Path = "runs", resume: str | Path | None = None) -> Path:
    if resume:
        config_path = Path(resume) / "config.toml"  # the run's own settings
    cfg = load_config(config_path)
    run_cfg = cfg["run"]
    registry = leaderboard.load_registry()
    specs = model_specs(run_cfg)
    players = {}
    for spec in specs:
        if not _answers(spec):
            print(f"not yet answering: {spec['name']} (port {spec['port']}); it joins when its server is up", flush=True)
            continue
        players[spec["name"]] = make_player(spec, registry.get(spec["name"], {}), run_cfg)
    if len(players) < run_cfg["num_players"]:
        raise SystemExit(f"only {len(players)} models answer; need at least {run_cfg['num_players']}")
    if resume:
        out = Path(resume)
        info = json.loads((out / "servers.json").read_text())
        _add_players(out, {n: p for n, p in players.items() if n not in info["servers"]})
    else:
        out = Path(out_root) / f"{datetime.now():%Y%m%d-%H%M%S}_{run_cfg['name']}"
        out.mkdir(parents=True)
        shutil.copy(config_path, out / "config.toml")
        _atomic(out / "servers.json", {"template_version": TEMPLATE_VERSION, "run": run_cfg, "servers": {}})
        _add_players(out, players)
    print(f"{len(players)} models: {', '.join(players)} -> {out}", flush=True)
    t = Tournament(cfg, out, players)
    if resume:
        t.restore()
    t.run_all()
    print(stats.to_markdown(stats.summarize(out)), flush=True)
    return out
