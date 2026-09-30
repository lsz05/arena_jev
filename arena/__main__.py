"""Command line.

    python -m arena run configs/sanity.toml [--deals N]     play a tournament
    python -m arena swiss configs/swiss.toml                Swiss-style tournament over the local model servers
    python -m arena stats runs/<dir>                        recompute a run's summary
    python -m arena prompt [--seed S --step K ...]          print the request a model would get
    python -m arena mock-server [--port 8199]               fake System One server for dry runs
    python -m arena viz runs/<dir> [--port 8200]            browse results and replay games
    python -m arena viz runs/<dir> --export out.html        one self-contained HTML file with a few replays
    python -m arena c4 <model-a> <model-b> [--openings 7]    Connect Four match between two players
    python -m arena c4-league [--openings 7]                 Connect Four league over every model server
    python -m arena c4-show runs/<dir> [--game G00-ab]       replay a Connect Four game in the terminal
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(prog="arena")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="play a tournament from a TOML config")
    r.add_argument("config")
    r.add_argument("--deals", type=int, help="override run.deals (games = deals x seat rotations)")
    r.add_argument("--out", default="runs")

    sw = sub.add_parser("swiss", help="Swiss-style tournament over the servers in servers/models/")
    sw.add_argument("config")
    sw.add_argument("--out", default="runs")
    sw.add_argument("--resume", metavar="RUN_DIR", help="continue a stopped swiss run (its own config.toml is used)")

    s = sub.add_parser("stats", help="recompute summary.json / summary.md for a run")
    s.add_argument("run_dir")

    p = sub.add_parser("prompt", help="print the request a model would receive at some point of a game")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--step", type=int, default=40, help="decision index in a heuristic-played game")
    p.add_argument("--template", default="B")
    p.add_argument("--history", type=int, default=5, help="history rounds (maximum)")
    p.add_argument("--max-tokens", type=int, default=512, help="token budget; oldest rounds are dropped to fit (0: none)")

    m = sub.add_parser("mock-server", help="serve a fake /v1/systemone endpoint")
    m.add_argument("--port", type=int, default=8199)

    si = sub.add_parser("site", help="serve the arena website (UNO viewer, leaderboard, bench)")
    si.add_argument("--uno-run", default="latest", help='run directory for /uno/, or "latest" (newest *_swiss run)')
    si.add_argument("--base", default="/arenas")
    si.add_argument("--host", default="127.0.0.1")
    si.add_argument("--port", type=int, default=8200)
    si.add_argument("--allowed-host", action="append", dest="allowed_hosts")

    v = sub.add_parser("viz", help="browse a run's results and replay its games")
    v.add_argument("run_dir")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8200)
    v.add_argument("--base-path", default="", help="also serve under this URL prefix, e.g. /arenas/uno")
    v.add_argument("--allowed-host", action="append", dest="allowed_hosts",
                   help="only answer requests for this Host (repeatable; local requests always allowed)")
    v.add_argument("--export", metavar="HTML", help="write a self-contained file instead of serving")
    v.add_argument("--deals", type=int, default=3, help="with --export: include replays of the first N deals")

    c4 = sub.add_parser("c4", help="Connect Four match between two players (model names, random, heuristic)")
    c4.add_argument("a")
    c4.add_argument("b")
    c4.add_argument("--openings", type=int, default=7, help="0: empty board only; 7: every first move; 49: every first two")
    c4.add_argument("--template", default="B", choices=["A", "B"])
    c4.add_argument("--policy", default="argmax", choices=["argmax", "sample"])
    c4.add_argument("--hint", choices=["defend"], help="append a strategy hint to the instructions (default: none)")
    c4.add_argument("--workers", type=int, default=4, help="games played at the same time")
    c4.add_argument("--seed", type=int, default=20260929)
    c4.add_argument("--out", default="runs")
    lg = sub.add_parser("c4-league", help="Connect Four league: every pair of model servers, every opening, both colors")
    lg.add_argument("--models", help="comma-separated model names (default: every servers/models/*.toml that answers)")
    lg.add_argument("--exclude", help="comma-separated model names to leave out")
    lg.add_argument("--openings", type=int, default=7, help="0, 7 or 49 openings per pair, each with both colors")
    lg.add_argument("--template", default="B", choices=["A", "B"])
    lg.add_argument("--policy", default="argmax", choices=["argmax", "sample"])
    lg.add_argument("--hint", choices=["defend"])
    lg.add_argument("--pair-workers", type=int, default=4, help="games of one pair played at the same time")
    lg.add_argument("--seed", type=int, default=20260929)
    lg.add_argument("--out", default="runs")
    lg.add_argument("--resume", metavar="RUN_DIR", help="continue a stopped league (its own settings are used)")
    pz = sub.add_parser("c4-puzzles-make", help="generate Connect Four decision puzzles (one correct column each)")
    pz.add_argument("--per-type", type=int, default=250)
    pz.add_argument("--seed", type=int, default=20261001)
    pz.add_argument("--out", default="data/c4-puzzles")
    pz.add_argument("--workers", type=int, default=16)
    pr = sub.add_parser("c4-puzzles-run", help="ask every model server every Connect Four puzzle in several option orders")
    pr.add_argument("--models", help="comma-separated model names (default: every servers/models/*.toml that answers)")
    pr.add_argument("--exclude", help="comma-separated model names to leave out")
    pr.add_argument("--orders", type=int, default=3)
    pr.add_argument("--per-model", type=int, default=4, help="requests in flight per model")
    pr.add_argument("--out", default="runs")
    pr.add_argument("--resume", metavar="RUN_DIR")
    c4s = sub.add_parser("c4-show", help="replay a Connect Four game in the terminal")
    c4s.add_argument("run_dir")
    c4s.add_argument("--game")

    a = ap.parse_args()
    if a.cmd == "c4":
        from .connect4.match import run as c4_run
        c4_run(a.a, a.b, a.openings, a.template, a.policy, a.workers, a.seed, a.out, hint=a.hint)
        return
    if a.cmd == "c4-league":
        from .connect4.league import run as league_run
        split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
        league_run(split(a.models), split(a.exclude), a.openings, a.template, a.policy, a.hint, a.pair_workers, a.seed,
                   a.out, a.resume)
        return
    if a.cmd == "c4-puzzles-make":
        from .connect4.puzzles import make
        print(json.dumps(make(a.per_type, a.seed, a.out, a.workers), indent=1))
        return
    if a.cmd == "c4-puzzles-run":
        from .connect4.puzzle_run import run as puzzles_run
        split = lambda v: [x.strip() for x in v.split(",") if x.strip()] if v else None  # noqa: E731
        puzzles_run(split(a.models), split(a.exclude), a.orders, a.per_model, a.out, a.resume)
        return
    if a.cmd == "c4-show":
        from .connect4.match import show
        print(show(a.run_dir, a.game))
        return
    if a.cmd == "run":
        from .tournament import run
        run(a.config, a.out, a.deals)
    elif a.cmd == "swiss":
        from .swiss import run as swiss_run
        swiss_run(a.config, a.out, resume=a.resume)
    elif a.cmd == "stats":
        config = Path(a.run_dir, "config.json")
        if config.exists() and json.loads(config.read_text()).get("league"):  # a Connect Four league
            from .connect4 import league
            print(league.standings(league.rebuild(a.run_dir)))
            return
        from . import stats
        summary = stats.summarize(a.run_dir)
        Path(a.run_dir, "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        Path(a.run_dir, "summary.md").write_text(stats.to_markdown(summary))
        from . import leaderboard
        leaderboard.write(a.run_dir)
        print(stats.to_markdown(summary))
    elif a.cmd == "prompt":
        show_prompt(a.seed, a.step, a.template, a.history, a.max_tokens)
    elif a.cmd == "site":
        from . import site
        site.serve(a.uno_run, a.base, a.host, a.port, a.allowed_hosts)
    elif a.cmd == "viz":
        from . import viz
        if a.export:
            viz.export(a.run_dir, a.export, a.deals)
        else:
            viz.serve(a.run_dir, a.host, a.port, a.base_path, a.allowed_hosts)
    elif a.cmd == "mock-server":
        from .mock_server import serve
        print(f"mock System One server on http://127.0.0.1:{a.port}")
        serve(a.port)


def show_prompt(seed: int, step: int, template: str, history: int, max_tokens: int) -> None:
    from . import tokens
    from .players import HeuristicPlayer
    from .uno import render
    from .uno.engine import START_COLOR, UnoGame

    game = UnoGame(4, seed)
    bot = HeuristicPlayer()
    while not game.over and game.steps < step:
        d = game.decision()
        game.step(bot.act(game.observation(d.player), d, random.Random(game.steps)).action)
    if game.over:
        raise SystemExit(f"the game ended before step {step}")
    d = game.decision()
    obs = game.observation(d.player)
    rng = random.Random(f"{seed}:{game.steps}")
    q = render.color_question(obs, d, None, template, rng) if d.kind == START_COLOR else render.move_question(obs, d, template, rng)
    count = tokens.counter() if max_tokens else None
    state, rounds, n = render.fit_state(obs, d, q, history, max_tokens or None, count)
    print(json.dumps({"model": "<model>", "state": state, "questions": {q.name: q.to_json()}}, indent=2, ensure_ascii=False))
    print(f"\n----- history kept: {rounds} of {history} rounds; budget tokens: {n} / {max_tokens or '-'} -----\n" + state)


if __name__ == "__main__":
    main()
