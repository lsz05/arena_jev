"""Run many games from a TOML config and write the results to runs/<timestamp>_<name>/.

Formats:
- "table": the listed players (exactly num_players) sit together.
- "gauntlet": each candidate plays against num_players - 1 copies of `opponent`.
Every deal is played once per seat rotation with the same deck, so each player gets each
starting hand once; `deals` = 250 with 4 players means 1000 games.

Output files: config.toml (copy), games.jsonl, decisions.jsonl, servers.json, summary.json, summary.md.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import leaderboard, stats, tokens
from .match import play_game
from .players import HeuristicPlayer, Player, RandomPlayer, SystemOnePlayer
from .systemone import SystemOneClient
from .uno.render import TEMPLATE_VERSION


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    run = cfg.setdefault("run", {})
    run.setdefault("name", Path(path).stem)
    run.setdefault("format", "table")
    run.setdefault("num_players", 4)
    run.setdefault("deals", 250)
    run.setdefault("seed", 0)
    run.setdefault("workers", 1)
    run.setdefault("template", "B")
    run.setdefault("history_rounds", 1)
    run.setdefault("policy", "argmax")
    # History is trimmed, oldest round first, only when a request would not fit the model's own input limit
    # (registry field max_input_tokens, counted with the model's own tokenizer). max_prompt_tokens is an
    # optional extra cap for every player.
    run.setdefault("max_prompt_tokens", None)
    run.setdefault("budget_tokenizer", tokens.DEFAULT_TOKENIZER)
    registry = leaderboard.load_registry()
    for spec in cfg["players"]:
        reg = registry.get(spec["name"], {})
        for key in ("tokenizer", "max_input_tokens"):
            if reg.get(key) is not None:
                spec.setdefault(key, reg[key])
    run.setdefault("log_decisions", True)
    run.setdefault("wd4_challenge", False)  # official Wild Draw Four challenge rule
    run.setdefault("shuffle_hand", False)  # list each player's hand in a seeded random order
    if not cfg.get("players"):
        raise ValueError("config needs at least one [[players]] entry")
    names = [p["name"] for p in cfg["players"]]
    if len(set(names)) != len(names):
        raise ValueError(f"player names must be unique: {names}")
    return cfg


def make_player(spec: dict, run: dict) -> Player:
    kind = spec.get("type", "systemone")
    if kind == "random":
        return RandomPlayer(spec["name"])
    if kind == "heuristic":
        return HeuristicPlayer(spec["name"])
    if kind == "systemone":
        key_env = spec.get("api_key_env")
        client = SystemOneClient(
            spec["base_url"],
            model=spec.get("model", "jev-latest"),
            api_key=os.environ.get(key_env) if key_env else None,
            timeout=spec.get("timeout", 60.0),
            retries=spec.get("retries", 2),
        )
        return SystemOnePlayer(
            spec["name"],
            client,
            template=spec.get("template", run["template"]),
            history_rounds=spec.get("history_rounds", run["history_rounds"]),
            policy=spec.get("policy", run["policy"]),
            max_options=spec.get("max_options", 255),
            max_prompt_tokens=_budget(spec, run),
            count_tokens=tokens.counter(spec.get("tokenizer") or run["budget_tokenizer"]) if _budget(spec, run) else None,
        )
    raise ValueError(f"unknown player type {kind!r}")


def _budget(spec: dict, run: dict) -> int | None:
    """Token budget for this player's requests: its own input limit and the run-wide cap, whichever is smaller."""
    limits = [x for x in (spec.get("max_input_tokens"), run.get("max_prompt_tokens")) if x]
    return min(limits) if limits else None


def schedule(cfg: dict, players: dict[str, Player]) -> list[tuple[str, int, int, list[Player]]]:
    """(game_id, seed, rotation, seating) for every game in the run."""
    run = cfg["run"]
    n = run["num_players"]
    lineups: list[list[Player]] = []
    if run["format"] == "table":
        if len(players) != n:
            raise ValueError(f"table format needs exactly {n} players, got {len(players)}")
        lineups.append(list(players.values()))
    elif run["format"] == "gauntlet":
        opponent = players[run["opponent"]]
        for name, p in players.items():
            if name != run["opponent"]:
                lineups.append([p] + [opponent] * (n - 1))
    else:
        raise ValueError(f"unknown format {run['format']!r}")
    games = []
    for li, lineup in enumerate(lineups):
        for deal in range(run["deals"]):
            seed = run["seed"] * 1_000_003 + deal
            for r in range(n):
                seating = lineup[r:] + lineup[:r]
                games.append((f"L{li}-D{deal}-R{r}", seed, r, seating))
    return games


def run(config_path: str | Path, out_root: str | Path = "runs", deals: int | None = None) -> Path:
    cfg = load_config(config_path)
    if deals is not None:
        cfg["run"]["deals"] = deals
    run_cfg = cfg["run"]
    players = {spec["name"]: make_player(spec, run_cfg) for spec in cfg["players"]}
    games = schedule(cfg, players)

    out = Path(out_root) / f"{datetime.now():%Y%m%d-%H%M%S}_{run_cfg['name']}"
    out.mkdir(parents=True)
    shutil.copy(config_path, out / "config.toml")
    servers = {name: {"base_url": p.client.base_url, "models": p.client.models()}
               for name, p in players.items() if isinstance(p, SystemOnePlayer)}
    (out / "servers.json").write_text(json.dumps(
        {"template_version": TEMPLATE_VERSION, "run": run_cfg, "servers": servers}, indent=2, ensure_ascii=False))

    lock = threading.Lock()
    games_f = open(out / "games.jsonl", "w")
    decisions_f = open(out / "decisions.jsonl", "w") if run_cfg["log_decisions"] else None

    def write_decision(record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with lock:
            decisions_f.write(line + "\n")

    def one(game_id: str, seed: int, rotation: int, seating: list[Player]) -> dict:
        return play_game(seating, seed, game_id=game_id, meta={"rotation": rotation},
                         on_decision=write_decision if decisions_f else None,
                         wd4_challenge=run_cfg["wd4_challenge"], shuffle_hand=run_cfg["shuffle_hand"],
                         must_play=run_cfg.get("must_play", False))

    start = time.time()
    done = 0
    print(f"{len(games)} games -> {out}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=run_cfg["workers"]) as pool:
            futures = [pool.submit(one, *g) for g in games]
            for fut in as_completed(futures):
                record = fut.result()
                with lock:
                    games_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1
                if done % max(1, len(games) // 20) == 0 or done == len(games):
                    rate = done / (time.time() - start)
                    print(f"  {done}/{len(games)} games  ({rate:.1f} games/s)", flush=True)
    finally:
        games_f.close()
        if decisions_f:
            decisions_f.close()

    summary = stats.summarize(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out / "summary.md").write_text(stats.to_markdown(summary))
    leaderboard.write(out)
    print(stats.to_markdown(summary))
    return out
