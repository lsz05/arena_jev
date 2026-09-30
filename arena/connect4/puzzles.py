"""Connect Four decision puzzles: positions with exactly one correct move, verified by the engine and the C solver.

    python -m arena c4-puzzles-make [--per-type 250] [--seed 20261001] [--out data/c4-puzzles]

Types (the mover is always "you"; the answer is one column):
    win     easy    one column wins at once, and the opponent has no immediate win
    block   medium  the opponent can win at one column next turn and the mover cannot win at once; blocking is the only
                    move that does not lose at once (and it does not let the opponent win on top of it)
    fork    hard    no immediate win or threat; exactly one move leaves threats the opponent cannot all stop (a win in
                    two moves), and the solver confirms it is the only winning move
    solver  hard    none of the above; the solver finds exactly one move that keeps the result (the only winning move,
                    or the only move that does not lose)

Positions come from random games (players avoid completing four when they can, so threats pile up). A position and
its left-right mirror count as the same puzzle, so no two puzzles are mirrors or copies of each other. Each type is
balanced over the answer column (and win / block over the direction of the four).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from . import render, solver
from .engine import COLS, Connect4Game

TYPES = {
    "win": ("easy", "One column wins at once; the opponent has no immediate win."),
    "block": ("medium", "The opponent can win at one column next turn and the mover cannot win at once: only the block "
                        "does not lose at once."),
    "fork": ("hard", "No immediate win or threat; exactly one move leaves threats the opponent cannot all stop, and it "
                     "is the only winning move."),
    "solver": ("hard", "No immediate win, threat or two-move win; exactly one move keeps the result (the only winning "
                       "move, or the only move that does not lose), by the solver."),
}
MIN_PLY_SOLVER = 10  # earlier positions take the solver from a second to minutes


def wins(g: Connect4Game, p: int) -> list[int]:
    return [c for c in g.legal() if g.wins_with(c, p)]


def line_of(g: Connect4Game, col: int, p: int) -> list[tuple[int, int]]:
    g.columns[col].append(p)
    try:
        return list(g._line_through(col, len(g.columns[col]) - 1, p))
    finally:
        g.columns[col].pop()


def direction(line) -> str:
    cols, rows = {c for c, _ in line}, {r for _, r in line}
    return "vertical" if len(cols) == 1 else "horizontal" if len(rows) == 1 else "diagonal"


def fork_moves(g: Connect4Game, p: int) -> list[int]:
    """Moves after which p wins next turn whatever the opponent replies (the opponent cannot win at once)."""
    out = []
    for c in g.legal():
        g2 = Connect4Game(g.moves + [c])
        if g2.over or wins(g2, 1 - p):
            continue
        replies = g2.legal()
        if replies and all(wins(Connect4Game(g2.moves + [r]), p) for r in replies):
            out.append(c)
    return out


def canonical(g: Connect4Game) -> str:
    cols = ["".join(str(v) for v in col) for col in g.columns]
    return min("|".join(cols), "|".join(reversed(cols)))


def classify(g: Connect4Game, analyze_share: float = 1.0, rng: random.Random | None = None) -> tuple[str, int, dict] | None:
    """(type, answer column 0-based, detail) when the position is a puzzle, else None. The solver (fork and solver
    types) runs on a share `analyze_share` of the quiet positions only, to save time."""
    if g.over:
        return None
    me, opp = g.to_move, 1 - g.to_move
    w, t = wins(g, me), wins(g, opp)
    if len(w) == 1 and not t:
        line = line_of(g, w[0], me)
        return "win", w[0], {"direction": direction(line), "line": [[c + 1, r + 1] for c, r in line]}
    if w:
        return None
    if len(t) == 1:
        g2 = Connect4Game(g.moves + [t[0]])
        if g2.over or wins(g2, opp):
            return None  # the block itself loses (the opponent wins on top of it)
        line = line_of(g, t[0], opp)
        return "block", t[0], {"direction": direction(line), "line": [[c + 1, r + 1] for c, r in line]}
    if t or g.ply < MIN_PLY_SOLVER or (rng is not None and rng.random() > analyze_share):
        return None
    values = solver.analyze(g.moves)
    legal_values = {c: v for c, v in enumerate(values) if v is not None}
    best = max(legal_values.values())
    keep = [c for c, v in legal_values.items() if (v > 0) - (v < 0) == (best > 0) - (best < 0)]
    outcome = {c + 1: ("win" if v > 0 else "draw" if v == 0 else "loss") for c, v in legal_values.items()}
    forks = fork_moves(g, me)
    if forks:
        if len(forks) == 1 and keep == forks and best > 0:
            g2 = Connect4Game(g.moves + [forks[0]])
            return "fork", forks[0], {
                "threats": [c + 1 for c in g2.legal() if g2.wins_with(c, me)],  # winning columns right after the move
                "replies": {str(r + 1): [c + 1 for c in wins(Connect4Game(g2.moves + [r]), me)] for r in g2.legal()},
                "outcome": outcome}
        return None
    if len(keep) == 1 and best >= 0:
        return "solver", keep[0], {"result": "win" if best > 0 else "draw", "outcome": outcome}
    return None


def _collect(args) -> list[dict]:
    seed, games = args
    rng = random.Random(seed)
    found = []
    for _ in range(games):
        g = Connect4Game()
        while not g.over:
            hit = classify(g, 0.5, rng)
            if hit:
                kind, col, detail = hit
                found.append({"type": kind, "moves": list(g.moves), "answer": col, "detail": detail,
                              "key": canonical(g), "ply": g.ply, "mover": g.to_move})
            quiet = [c for c in g.legal() if not g.wins_with(c, g.to_move)] or g.legal()
            g.play(rng.choice(quiet))
    return found


def _balanced(cands: list[dict], n: int, by_direction: bool) -> list[dict]:
    """n candidates spread evenly over answer columns (and directions); deterministic given the candidates."""
    groups = defaultdict(list)
    for c in cands:
        groups[(c["answer"], c["detail"].get("direction") if by_direction else "")].append(c)
    for g in groups.values():
        g.sort(key=lambda c: hashlib.sha1(c["key"].encode()).hexdigest())
    keys = sorted(groups)
    picked, round_ = [], 0
    while len(picked) < n and any(round_ < len(groups[k]) for k in keys):
        by_col = defaultdict(list)
        for k in keys:
            if round_ < len(groups[k]):
                by_col[k[0]].append(groups[k][round_])
        for col in sorted(by_col):  # one per (column, direction) per round, columns in turn
            for c in by_col[col]:
                if len(picked) < n:
                    picked.append(c)
        round_ += 1
    return picked


def make(per_type: int = 250, seed: int = 20261001, out: str | Path = "data/c4-puzzles", workers: int = 16,
         batch: int = 400) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pool_of: dict[str, dict[str, dict]] = {k: {} for k in TYPES}
    seen: set[str] = set()
    step = 0
    with Pool(workers) as pool:
        while True:
            jobs = [(seed * 1000 + step * workers + i, batch) for i in range(workers)]
            step += 1
            for found in pool.map(_collect, jobs):
                for f in found:
                    if f["key"] not in seen:
                        seen.add(f["key"])
                        pool_of[f["type"]][f["key"]] = f
            counts = {k: _column_counts(v.values()) for k, v in pool_of.items()}
            need = per_type // COLS + 1
            print(f"round {step}: " + ", ".join(f"{k} {len(v)} (fewest per column {min(counts[k]) if counts[k] else 0})"
                                                for k, v in pool_of.items()), flush=True)
            if all(len(counts[k]) == COLS and min(counts[k]) >= need * (3 if k in ("win", "block") else 1) for k in TYPES):
                break
    summary = {}
    for kind, (tier, rule) in TYPES.items():
        chosen = _balanced(list(pool_of[kind].values()), per_type, by_direction=kind in ("win", "block"))
        rows = [item(kind, i, c, tier, seed) for i, c in enumerate(chosen)]
        (out / f"{kind}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        summary[kind] = {"items": len(rows), "tier": tier, "rule": rule, "candidates": len(pool_of[kind]),
                         "by_answer_column": _column_counts(chosen)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _column_counts(cands) -> list[int]:
    counts = [0] * COLS
    for c in cands:
        counts[c["answer"]] += 1
    return counts if any(counts) else []


def item(kind: str, i: int, c: dict, tier: str, seed: int) -> dict:
    g = Connect4Game(c["moves"])
    d = g.decision()
    q = render.question(g, d, "B")  # options in column order; the runner shuffles them
    return {
        "id": f"c4-{kind}-{i:03d}", "family": kind, "tier": tier, "moves": "".join(str(m + 1) for m in c["moves"]),
        "ply": c["ply"], "mover": "first" if c["mover"] == 0 else "second",
        "state": render.render_state(g, d), "question": q.to_json(), "labels": list(q.options),
        "expected": str(c["answer"] + 1), "detail": c["detail"],
        "provenance": {"source": f"random games, seed {seed}", "verified": "Connect Four engine" + (
            " and the C solver" if kind in ("fork", "solver") else ""), "license": "generated by this project"},
    }
