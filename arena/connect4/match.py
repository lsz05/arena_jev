"""Play Connect Four games, log every decision, and run a match between two players.

    python -m arena c4 kev-0.8b decider-0.8b                     # 7 openings x both colors = 14 games
    python -m arena c4 kev-0.8b heuristic --openings 49 --workers 8
    python -m arena c4-show runs/<dir> [--game G00-3-1]          # replay a game in the terminal

A player is a model file name in servers/models/ (its server must be running), or "random" / "heuristic".
Output: runs/<time>_c4-<a>-vs-<b>/ with games.jsonl, decisions.jsonl, summary.json and summary.md.
"""

from __future__ import annotations

import itertools
import json
import random
import threading
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .. import tokens
from ..systemone import SystemOneClient
from . import render
from .engine import COLS, Connect4Game
from .players import HeuristicPlayer, RandomPlayer, SystemOnePlayer, gives_win_on_top

ROOT = Path(__file__).resolve().parents[2]


# ---- one game -------------------------------------------------------------------------------------


def play_game(players, opening, seed: int, game_id: str, on_decision=None, meta: dict | None = None,
              template: str = "B") -> dict:
    """players[0] moves first (after the opening, whose moves alternate from player 0)."""
    game = Connect4Game(opening)
    while not game.over:
        decision = game.decision()
        me, opp = decision.player, 1 - decision.player
        wins = [c for c in decision.legal if game.wins_with(c, me)]
        threats = [c for c in decision.legal if game.wins_with(c, opp)]
        rng = random.Random(f"{seed}:{game.ply}")
        choice = players[me].act(game, decision, rng)
        col = choice.col
        diag = {
            "could_win": bool(wins), "took_win": col in wins,
            "must_block": bool(threats) and not wins, "blocked": col in threats,
            "gave_win_on_top": gives_win_on_top(game, col, me) and not game.wins_with(col, me),
        }
        record = {"game": game_id, "ply": game.ply, "seat": me, "player": players[me].name,
                  "legal": [c + 1 for c in decision.legal], "action": col + 1, "diag": diag, **choice.info}
        game.play(col)
        if on_decision:
            on_decision(record)
    names = [p.name for p in players]
    return {
        "game": game_id, "seed": seed, "players": names, "opening": [c + 1 for c in game.opening],
        "moves": [c + 1 for c in game.moves], "winner_seat": game.winner,
        "winner": names[game.winner] if game.winner is not None else None,
        "result": "draw" if game.winner is None else ("first" if game.winner == 0 else "second"),
        "plies": game.ply, "line": [[c + 1, r + 1] for c, r in game.line],
        "rules": {"template": template}, **(meta or {}),
    }


# ---- players by name ------------------------------------------------------------------------------


def make_player(name: str, template: str = "B", policy: str = "argmax"):
    if name == "random":
        return RandomPlayer()
    if name == "heuristic":
        return HeuristicPlayer()
    path = ROOT / "servers" / "models" / f"{name}.toml"
    if not path.exists():
        raise SystemExit(f"unknown player {name!r}: not random / heuristic, and no {path}")
    spec = tomllib.loads(path.read_text())
    reg = next((m for m in json.loads((ROOT / "bench" / "models.json").read_text())["models"] if m["name"] == name), {})
    limit = spec.get("max_input_tokens") or reg.get("max_input_tokens")
    tok = spec.get("tokenizer") or reg.get("tokenizer") or tokens.DEFAULT_TOKENIZER
    client = SystemOneClient(f"http://127.0.0.1:{spec['port']}", model=name, timeout=120, retries=2)
    return SystemOnePlayer(name, client, template=template, policy=policy,
                           max_options=spec.get("max_options") or reg.get("max_options") or 255,
                           max_prompt_tokens=limit, count_tokens=tokens.counter(tok) if limit else None)


def openings(n: int) -> list[tuple[int, ...]]:
    """0: the empty board only; 7: every first move; 49: every pair of first moves (0-based columns)."""
    if n == 0:
        return [()]
    if n == 7:
        return [(c,) for c in range(COLS)]
    if n == 49:
        return list(itertools.product(range(COLS), repeat=2))
    raise SystemExit("--openings must be 0, 7 or 49")


# ---- a match --------------------------------------------------------------------------------------


def run(a: str, b: str, n_openings: int = 7, template: str = "B", policy: str = "argmax", workers: int = 4,
        seed: int = 20260929, out_root: str | Path = "runs") -> Path:
    pa, pb = make_player(a, template, policy), make_player(b, template, policy)
    if pa.name == pb.name:
        raise SystemExit("the two players must differ")
    out = Path(out_root) / f"{datetime.now():%Y%m%d-%H%M%S}_c4-{a}-vs-{b}"
    out.mkdir(parents=True)
    config = {"a": a, "b": b, "openings": n_openings, "template": template, "policy": policy, "seed": seed}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    schedule = []
    for i, op in enumerate(openings(n_openings)):
        for first, second in ((pa, pb), (pb, pa)):
            gid = f"G{i:02d}-{'ab' if first is pa else 'ba'}"
            schedule.append((gid, (first, second), op, seed * 1000 + i))
    lock = threading.Lock()
    dec_f, games_f = open(out / "decisions.jsonl", "w"), open(out / "games.jsonl", "w")

    def log(d):
        with lock:
            dec_f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def one(item):
        gid, players, op, s = item
        g = play_game(list(players), op, s, gid, on_decision=log, template=template)
        with lock:
            games_f.write(json.dumps(g, ensure_ascii=False) + "\n")
            games_f.flush()
        return g

    print(f"{a} vs {b}: {len(schedule)} games ({n_openings or 'no'} openings x both colors) -> {out}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        games = list(pool.map(one, schedule))
    dec_f.close()
    games_f.close()
    summary = summarize(out)
    print(to_markdown(summary), flush=True)
    return out


def summarize(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    games = [json.loads(line) for line in open(run_dir / "games.jsonl")]
    decisions = [json.loads(line) for line in open(run_dir / "decisions.jsonl")]
    per = defaultdict(lambda: Counter())
    for g in games:
        for seat, name in enumerate(g["players"]):
            side = "first" if seat == 0 else "second"
            res = "draw" if g["winner_seat"] is None else ("win" if g["winner_seat"] == seat else "loss")
            per[name][res] += 1
            per[name][f"{res}_{side}"] += 1
            per[name]["games"] += 1
    calls = defaultdict(lambda: {"decisions": 0, "calls": 0, "fallbacks": 0, "latency": 0.0, "pos": Counter()})
    diag = defaultdict(lambda: Counter())
    for d in decisions:
        name, c = d["player"], calls[d["player"]]
        c["decisions"] += 1
        c["fallbacks"] += bool(d.get("fallback"))
        for call in d.get("calls") or []:
            c["calls"] += 1
            c["latency"] += call["latency_ms"]
            c["pos"][call["options"].index(call["pick"]) if len(call["options"]) == 7 else "n/a"] += 1
        x = d["diag"]
        diag[name]["could_win"] += x["could_win"]
        diag[name]["took_win"] += x["took_win"] and x["could_win"]
        diag[name]["must_block"] += x["must_block"]
        diag[name]["blocked"] += x["blocked"] and x["must_block"]
        diag[name]["gave_win_on_top"] += x["gave_win_on_top"]
    rate = lambda k, n: round(k / n, 3) if n else None  # noqa: E731
    players = {}
    for name, r in per.items():
        c, x = calls[name], diag[name]
        players[name] = {
            "games": r["games"], "wins": r["win"], "draws": r["draw"], "losses": r["loss"],
            "score": rate(r["win"] + 0.5 * r["draw"], r["games"]),
            "as_first": {"wins": r["win_first"], "draws": r["draw_first"], "losses": r["loss_first"]},
            "as_second": {"wins": r["win_second"], "draws": r["draw_second"], "losses": r["loss_second"]},
            "took_immediate_win": rate(x["took_win"], x["could_win"]), "chances_to_win": x["could_win"],
            "blocked_immediate_threat": rate(x["blocked"], x["must_block"]), "threats_to_block": x["must_block"],
            "let_opponent_win_on_top": rate(x["gave_win_on_top"], c["decisions"]),
            "decisions": c["decisions"], "model_calls": c["calls"], "fallbacks": c["fallbacks"],
            "avg_latency_ms": round(c["latency"] / c["calls"], 1) if c["calls"] else None,
            "pick_position_7": {str(k): v for k, v in sorted(c["pos"].items(), key=lambda kv: str(kv[0])) if k != "n/a"},
        }
    summary = {"games": len(games), "avg_plies": round(sum(g["plies"] for g in games) / max(1, len(games)), 1),
               "results": Counter(g["result"] for g in games), "players": players}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "summary.md").write_text(to_markdown(summary))
    return summary


def to_markdown(s: dict) -> str:
    pct = lambda v: "–" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"{s['games']} games, {s['avg_plies']} plies on average; results {dict(s['results'])}", "",
             "| Player | W-D-L | Score | as first | as second | took win | blocked | let win on top | fallbacks | latency |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, p in s["players"].items():
        f, sc = p["as_first"], p["as_second"]
        lines.append(f"| {name} | {p['wins']}-{p['draws']}-{p['losses']} | {pct(p['score'])} | "
                     f"{f['wins']}-{f['draws']}-{f['losses']} | {sc['wins']}-{sc['draws']}-{sc['losses']} | "
                     f"{pct(p['took_immediate_win'])} of {p['chances_to_win']} | {pct(p['blocked_immediate_threat'])} of "
                     f"{p['threats_to_block']} | {pct(p['let_opponent_win_on_top'])} | {p['fallbacks']} | "
                     f"{'–' if p['avg_latency_ms'] is None else str(round(p['avg_latency_ms'])) + ' ms'} |")
    return "\n".join(lines)


# ---- terminal replay ------------------------------------------------------------------------------


def show(run_dir: str | Path, game_id: str | None = None) -> str:
    run_dir = Path(run_dir)
    games = {json.loads(line)["game"]: json.loads(line) for line in open(run_dir / "games.jsonl")}
    gid = game_id or sorted(games)[0]
    g = games[gid]
    decisions = sorted((json.loads(line) for line in open(run_dir / "decisions.jsonl") if f'"game": "{gid}"' in line),
                       key=lambda d: d["ply"])
    by_ply = {d["ply"]: d for d in decisions}
    game = Connect4Game()
    out = [f"{gid}: {g['players'][0]} (first) vs {g['players'][1]} (second); opening {g['opening']}; "
           f"winner {g['winner'] or 'none (draw)'}"]
    for ply, col in enumerate(g["moves"]):
        d = by_ply.get(ply)
        who = g["players"][ply % 2]
        if d is None:
            note = "opening move"
        elif d.get("calls"):
            probs = d["calls"][-1]["probabilities"]
            note = "p: " + " ".join(f"{k}={v:.2f}" for k, v in sorted(probs.items()))
            note += " (fallback)" if d.get("fallback") else ""
            note += " MISSED WIN" if d["diag"]["could_win"] and not d["diag"]["took_win"] else ""
            note += " MISSED BLOCK" if d["diag"]["must_block"] and not d["diag"]["blocked"] else ""
        else:
            note = "forced" if d.get("forced") else ""
        game.play(col - 1)
        out.append(f"\nply {ply + 1}: {who} plays column {col}   {note}\n" + render.board_text(game, 0).replace("X", "x").replace("O", "o"))
    return "\n".join(out) + "\n(x = first player, o = second player)"
