"""A Connect Four league: every pair of models plays every opening with both colors.

    python -m arena c4-league                                  # all servers in servers/models/ that answer
    python -m arena c4-league --openings 7 --pair-workers 4
    python -m arena c4-league --resume runs/<time>_c4-league   # continue a stopped league

A model plays one pair at a time (its games in that pair run in parallel, at most --pair-workers at once, which bounds
the concurrent requests to its server). Free models are paired in the order that keeps the models with the most
pairs left busy, since the slowest model's queue decides when the league ends. A pair's games are recorded together
when the pair ends; decisions are logged as they are made (on resume, decisions of unrecorded pairs are dropped).

Standings: points (win 1, draw 0.5) per game played, plus TrueSkill from the games in the order they were recorded.
Output: runs/<time>_c4-league/ with games.jsonl, decisions.jsonl, status.json, summary.json and standings.md.
"""

from __future__ import annotations

import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..leaderboard import rating_env
from ..swiss import _answers, model_specs
from .match import make_player, openings, play_game, summarize


def _atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


class League:
    def __init__(self, out: Path, players: dict, config: dict):
        names = list(players)
        self.out, self.names, self.config, self.players = out, names, config, players
        self.pairs = list(itertools.combinations(sorted(names), 2))
        self.done: set[tuple[str, str]] = set()
        self.busy: set[str] = set()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.status_lock = threading.Lock()
        self.summarized = 0.0
        self.env = rating_env(draw_probability=0.05)
        self.ratings = {n: self.env.create_rating() for n in names}
        self.games = 0
        self.started = time.time()

    # ---- resume -------------------------------------------------------------------------------------

    def restore(self) -> None:
        games = [json.loads(line) for line in open(self.out / "games.jsonl") if line.strip()]
        per_pair: dict[tuple[str, str], int] = {}
        for g in games:
            key = tuple(sorted(g["players"]))
            per_pair[key] = per_pair.get(key, 0) + 1
            self._rate(g)
        per = 2 * len(openings(self.config["openings"]))
        self.done = {p for p, n in per_pair.items() if n >= per}
        self.games = len(games)
        ids = {g["game"] for g in games}
        kept = [line for line in open(self.out / "decisions.jsonl") if json.loads(line)["game"] in ids]
        (self.out / "decisions.jsonl").write_text("".join(kept))
        print(f"resumed: {len(self.done)} of {len(self.pairs)} pairs done, {self.games} games", flush=True)

    # ---- scheduling ---------------------------------------------------------------------------------

    def left(self, name: str) -> int:
        return sum(1 for p in self.pairs if name in p and p not in self.done and not (set(p) <= self.busy))

    def pick(self) -> tuple[str, str] | None:
        free = [p for p in self.pairs if p not in self.done and p[0] not in self.busy and p[1] not in self.busy]
        if not free:
            return None
        return max(free, key=lambda p: (self.left(p[0]) + self.left(p[1]), max(self.left(p[0]), self.left(p[1]))))

    def play_pair(self, pair: tuple[str, str], index: int) -> None:
        a, b = pair
        schedule = []
        for i, op in enumerate(openings(self.config["openings"])):
            for first, second, tag in ((a, b, "ab"), (b, a, "ba")):
                schedule.append((f"P{index:03d}-G{i:02d}-{tag}", [self.players[first], self.players[second]], op,
                                 self.config["seed"] * 1000 + i))

        def one(item):
            gid, players, op, seed = item
            return play_game(players, op, seed, gid, on_decision=self.log, template=self.config["template"],
                             hint=self.config["hint"], meta={"pair": index})

        results = []
        try:
            with ThreadPoolExecutor(max_workers=self.config["pair_workers"]) as pool:
                results = list(pool.map(one, schedule))
        except Exception as e:  # noqa: BLE001 - a failed pair is retried on resume, the league goes on
            print(f"pair {a} vs {b} failed: {type(e).__name__}: {e}", flush=True)
        with self.cond:
            if results:
                with open(self.out / "games.jsonl", "a") as f:
                    f.write("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in results))
                for g in results:
                    self._rate(g)
                self.games += len(results)
                self.done.add(pair)
            self.busy.difference_update(pair)
            self.cond.notify_all()
        self.write_status()  # outside the lock: the summary reads the whole log

    def log(self, d: dict) -> None:
        line = json.dumps(d, ensure_ascii=False) + "\n"
        with self.lock:
            with open(self.out / "decisions.jsonl", "a") as f:
                f.write(line)

    def _rate(self, g: dict) -> None:
        rate_game(self.env, self.ratings, g)

    # ---- output -------------------------------------------------------------------------------------

    def write_status(self, state: str = "running") -> None:
        with self.lock:
            status = {"state": state, "started": datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M:%S"),
                      "elapsed_s": round(time.time() - self.started), "pairs_done": len(self.done),
                      "pairs": len(self.pairs), "games": self.games, "busy": sorted(self.busy)}
            ratings = dict(self.ratings)
        with self.status_lock:
            _atomic(self.out / "status.json", status)
            if not status["games"] or (state == "running" and time.time() - self.summarized < 30):
                return
            self.summarized = time.time()
            write_summary(self.out, ratings, status)

    def run(self) -> None:
        self.write_status()  # before taking the lock below
        index = {p: i for i, p in enumerate(self.pairs)}
        last = time.time()
        with ThreadPoolExecutor(max_workers=len(self.names) // 2 + 1) as pool, self.cond:
            while len(self.done) < len(self.pairs):
                pair = self.pick()
                if pair is None:
                    if not self.busy:
                        print("no pair can be played and nothing is running; stopping", flush=True)
                        break
                    self.cond.wait(timeout=30)
                else:
                    self.busy.update(pair)
                    pool.submit(self.play_pair, pair, index[pair])
                if time.time() - last > 300:
                    last = time.time()
                    print(f"[{datetime.now():%H:%M:%S}] {len(self.done)}/{len(self.pairs)} pairs, {self.games} games", flush=True)
            while self.busy:
                self.cond.wait(timeout=30)
        self.write_status("done")
        print(standings(json.loads((self.out / "summary.json").read_text())), flush=True)


def rate_game(env, ratings: dict, g: dict) -> None:
    a, b = g["players"]
    if a not in ratings or b not in ratings:
        return
    ranks = [0, 0] if g["winner_seat"] is None else ([0, 1] if g["winner_seat"] == 0 else [1, 0])
    (ra,), (rb,) = env.rate([(ratings[a],), (ratings[b],)], ranks=ranks)
    ratings[a], ratings[b] = ra, rb


def write_summary(out: Path, ratings: dict, status: dict) -> dict:
    """summary.json (the match summary plus each model's TrueSkill and the league status) and standings.md."""
    summary = summarize(out)  # also writes summary.json / summary.md
    for name, p in summary["players"].items():
        r = ratings.get(name)
        if r is not None:
            p.update(ts_mu=round(r.mu, 3), ts_sigma=round(r.sigma, 3), ts_score=round(r.mu - 3 * r.sigma, 3))
    summary["status"] = status
    _atomic(out / "summary.json", summary)
    (out / "standings.md").write_text(standings(summary))
    return summary


def rebuild(out: str | Path) -> dict:
    """Recompute a league's ratings and summary from its recorded games (`python -m arena stats <run>`)."""
    out = Path(out)
    games = [json.loads(line) for line in open(out / "games.jsonl") if line.strip()]
    env = rating_env(draw_probability=0.05)
    ratings = {n: env.create_rating() for g in games for n in g["players"]}
    for g in games:
        rate_game(env, ratings, g)
    return write_summary(out, ratings, json.loads((out / "status.json").read_text()))


def standings(summary: dict) -> str:
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    rows = sorted(summary["players"].items(), key=lambda kv: (-(kv[1]["score"] or 0), -(kv[1].get("ts_score") or 0)))
    lines = [f"{summary['games']} games, {summary['avg_plies']} plies on average", "",
             "| # | Model | Games | W-D-L | Points | TrueSkill | took win | blocked | p on block (chance) | fallbacks |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (name, p) in enumerate(rows, 1):
        lines.append(f"| {i} | {name} | {p['games']} | {p['wins']}-{p['draws']}-{p['losses']} | {pct(p['score'])} | "
                     f"{p.get('ts_score', '–')} | {pct(p['took_immediate_win'])} | {pct(p['blocked_immediate_threat'])} | "
                     f"{p['p_on_block']} ({p['p_on_block_chance']}) | {p['fallbacks']} |")
    return "\n".join(lines)


def run(models: list[str] | None = None, exclude: list[str] | None = None, n_openings: int = 7, template: str = "B",
        policy: str = "argmax", hint: str | None = None, pair_workers: int = 4, seed: int = 20260929,
        out_root: str | Path = "runs", resume: str | Path | None = None) -> Path:
    if resume:
        out = Path(resume)
        config = json.loads((out / "config.json").read_text())
    else:
        config = {"league": True, "openings": n_openings, "template": template, "policy": policy, "hint": hint,
                  "pair_workers": pair_workers, "seed": seed}
    specs = model_specs({"models": models or [], "exclude": exclude or []})
    names = []
    for spec in specs:
        if _answers(spec):
            names.append(spec["name"])
        else:
            print(f"skipping {spec['name']}: its server on port {spec['port']} does not answer", flush=True)
    if len(names) < 2:
        raise SystemExit("fewer than two model servers answer")
    if not resume:
        out = Path(out_root) / f"{datetime.now():%Y%m%d-%H%M%S}_c4-league{'-' + hint if hint else ''}"
        out.mkdir(parents=True)
        config["players"] = names
        (out / "config.json").write_text(json.dumps(config, indent=1))
        (out / "games.jsonl").touch()
        (out / "decisions.jsonl").touch()
    players = {n: make_player(n, config["template"], config["policy"], config["hint"]) for n in names}
    league = League(out, players, config)
    if resume:
        league.restore()
    per = 2 * len(openings(config["openings"]))
    print(f"league: {len(names)} models, {len(league.pairs)} pairs x {per} games = {len(league.pairs) * per} games -> {out}",
          flush=True)
    league.run()
    return out
