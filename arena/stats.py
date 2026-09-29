"""Summaries of a run directory: win rates with confidence intervals, points, and per-decision diagnostics."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_jsonl(path: Path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


class Aggregator:
    """Running totals over game and decision records; `summary()` gives the same shape as `summarize()`."""

    def __init__(self):
        self.games = 0
        self.aborted = 0
        self.turns = 0
        self.per = defaultdict(lambda: {"seat_games": 0, "wins": 0, "points_won": 0, "points_left": 0})
        self.per_seat = defaultdict(lambda: [0, 0])  # seat -> [wins, games]
        self.diag = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # player -> flag -> [true, total]
        self.calls = defaultdict(lambda: {"decisions": 0, "forced": 0, "calls": 0, "fallbacks": 0, "latency_ms": 0.0,
                                          "input_tokens": 0, "rounds": 0, "rounds_n": 0})

    def add_game(self, g: dict) -> None:
        self.games += 1
        self.aborted += bool(g["aborted"])
        self.turns += g["turns"]
        for s, name in enumerate(g["seats"]):
            p = self.per[name]
            p["seat_games"] += 1
            p["points_left"] += g["hand_points"][s]
            won = g["winner_seat"] == s
            p["wins"] += won
            p["points_won"] += g["score"] if won else 0
            self.per_seat[s][0] += won
            self.per_seat[s][1] += 1

    def add_decision(self, d: dict) -> None:
        for flag, value in d.get("diag", {}).items():
            if value is None:
                continue
            self.diag[d["player"]][flag][0] += bool(value)
            self.diag[d["player"]][flag][1] += 1
        c = self.calls[d["player"]]
        c["decisions"] += 1
        c["fallbacks"] += bool(d.get("fallback"))
        c["forced"] += bool(d.get("forced"))
        for call in d.get("calls", []):
            c["calls"] += 1
            c["latency_ms"] += call.get("latency_ms") or 0.0
            c["input_tokens"] += call.get("input_tokens") or 0
            if call.get("history_rounds") is not None:
                c["rounds"] += call["history_rounds"]
                c["rounds_n"] += 1

    def summary(self) -> dict:
        players = {}
        for name, p in sorted(self.per.items()):
            n = p["seat_games"]
            lo, hi = wilson(p["wins"], n)
            entry = {
                "seat_games": n,
                "wins": p["wins"],
                "win_rate": p["wins"] / n if n else 0.0,
                "win_rate_ci95": [lo, hi],
                "avg_points_won": p["points_won"] / n if n else 0.0,
                "avg_points_left": p["points_left"] / n if n else 0.0,
                "diagnostics": {flag: {"rate": t / total, "n": total} for flag, (t, total) in sorted(self.diag[name].items())},
            }
            c = self.calls.get(name)
            if c and c["calls"]:
                entry["model_calls"] = {
                    "decisions": c["decisions"],
                    "forced": c["forced"],
                    "calls": c["calls"],
                    "fallback_rate": c["fallbacks"] / max(1, c["decisions"] - c["forced"]),
                    "avg_latency_ms": c["latency_ms"] / c["calls"],
                    "avg_input_tokens": c["input_tokens"] / c["calls"],
                    "avg_history_rounds": c["rounds"] / c["rounds_n"] if c["rounds_n"] else None,
                }
            players[name] = entry
        return {
            "games": self.games,
            "aborted": self.aborted,
            "avg_turns": self.turns / self.games if self.games else 0.0,
            "players": players,
            "seat_win_rate": {f"P{s}": w / n for s, (w, n) in sorted(self.per_seat.items())},
        }


def summarize(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    agg = Aggregator()
    for g in _read_jsonl(run_dir / "games.jsonl"):
        agg.add_game(g)
    for d in _read_jsonl(run_dir / "decisions.jsonl"):
        agg.add_decision(d)
    return agg.summary()


def to_markdown(summary: dict) -> str:
    lines = [
        f"games: {summary['games']}  (aborted: {summary['aborted']}, avg turns: {summary['avg_turns']:.1f})",
        "",
        "| player | seat-games | wins | win rate | 95% CI | avg points won | avg points left | drew with playable | wild with colored | fallback |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, p in sorted(summary["players"].items(), key=lambda kv: -kv[1]["win_rate"]):
        d = p["diagnostics"]
        rate = lambda k: f"{d[k]['rate']:.1%}" if k in d else "-"  # noqa: E731
        fb = f"{p['model_calls']['fallback_rate']:.1%}" if "model_calls" in p else "-"
        lo, hi = p["win_rate_ci95"]
        lines.append(
            f"| {name} | {p['seat_games']} | {p['wins']} | {p['win_rate']:.1%} | {lo:.1%}–{hi:.1%} | "
            f"{p['avg_points_won']:.1f} | {p['avg_points_left']:.1f} | {rate('drew_with_playable')} | "
            f"{rate('wild_with_colored')} | {fb} |"
        )
    lines += ["", "seat win rate: " + ", ".join(f"{s} {r:.1%}" for s, r in summary["seat_win_rate"].items())]
    models = {n: p["model_calls"] for n, p in summary["players"].items() if "model_calls" in p}
    if models:
        lines += ["", "| player | calls per real decision | avg latency ms | avg input tokens |", "|---|---|---|---|"]
        for n, m in sorted(models.items()):
            lines.append(f"| {n} | {m['calls'] / max(1, m['decisions'] - m['forced']):.2f} | {m['avg_latency_ms']:.0f} | {m['avg_input_tokens']:.0f} |")
    return "\n".join(lines) + "\n"
