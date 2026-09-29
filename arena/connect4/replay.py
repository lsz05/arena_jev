"""Replay data for the Connect Four page of the site: the matches in runs/, their games, and one game step by step.

A frame is one move: the board before it, the legal columns, the columns that would win at once for the mover
("wins") or for the opponent ("threats"), the move played, and for a model's move its calls (probabilities) and the
request it was sent, rebuilt from the game's seed (checked against the logged option order).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from . import render
from .engine import Connect4Game


def list_runs(root: Path) -> list[dict]:
    out = []
    for d in sorted(root.glob("*_c4-*"), reverse=True):
        if not (d / "games.jsonl").exists():
            continue
        config = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
        summary = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
        if summary is None:
            from .match import summarize  # a run still in progress, or stopped before its summary
            summary = summarize(d)
        out.append({"run": d.name, "a": config.get("a"), "b": config.get("b"), "openings": config.get("openings"),
                    "template": config.get("template"), "policy": config.get("policy"),
                    "games": summary["games"], "avg_plies": summary["avg_plies"], "players": summary["players"]})
    return out


def games(run_dir: Path) -> list[dict]:
    keys = ("game", "players", "opening", "winner", "winner_seat", "result", "plies")
    return sorted(({k: g.get(k) for k in keys} for g in map(json.loads, open(run_dir / "games.jsonl"))),
                  key=lambda g: g["game"])


def frames(run_dir: Path, game_id: str) -> dict:
    record = next((g for g in map(json.loads, open(run_dir / "games.jsonl")) if g["game"] == game_id), None)
    if record is None:
        raise KeyError(game_id)
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    template = (record.get("rules") or {}).get("template") or config.get("template", "B")
    marker = f'"game": "{game_id}"'
    by_ply = {d["ply"]: d for d in map(json.loads, (line for line in open(run_dir / "decisions.jsonl") if marker in line))}
    names = record["players"]
    game = Connect4Game()
    out = []
    for ply, col in enumerate(record["moves"]):
        decision = game.decision()
        me, opp = decision.player, 1 - decision.player
        frame = {
            "ply": ply, "seat": me, "player": names[me],
            "board": [list(c) for c in game.columns],
            "legal": [c + 1 for c in decision.legal],
            "wins": [c + 1 for c in decision.legal if game.wins_with(c, me)],
            "threats": [c + 1 for c in decision.legal if game.wins_with(c, opp)],
            "action": col, "landing_row": game.landing_row(col - 1) + 1,
            "opening": ply < len(record["opening"]),
        }
        d = by_ply.get(ply)
        if d is not None:
            for k in ("calls", "forced", "fallback", "error", "diag"):
                if k in d:
                    frame[k] = d[k]
            if d.get("calls"):
                q = render.question(game, decision, template, random.Random(f"{record['seed']}:{ply}"))
                first = d["calls"][0]
                state = render.render_state(game, decision, with_moves=first.get("with_moves", True))
                frame["request"] = {"state": state, "questions": {q.name: q.to_json()}}
                frame["matches_log"] = list(q.options)[:len(first["options"])] == first["options"]
                frame["knockout"] = len(d["calls"]) > 1
        out.append(frame)
        game.play(col - 1)
    if game.winner != record["winner_seat"]:
        raise ValueError(f"{game_id}: replay winner {game.winner} != logged {record['winner_seat']}")
    end = {"ply": game.ply, "end": True, "board": [list(c) for c in game.columns],
           "winner": record["winner"], "winner_seat": record["winner_seat"],
           "line": [[c + 1, r + 1] for c, r in game.line]}
    meta = {k: record.get(k) for k in ("game", "players", "opening", "winner", "winner_seat", "result", "plies", "seed")}
    meta["template"] = template
    return {"meta": meta, "frames": out + [end]}
