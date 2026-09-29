"""Leaderboard: one row per model with its exact size, UNO metrics, TrueSkill rating and JevBench accuracy.

Sources:
- the run directory: games.jsonl (for TrueSkill) and summary.json (win rates, points, diagnostics);
- the model registry bench/models.json (display name, repo, parameter count, languages);
- JevBench runs runs/jevbench-*/<model>/ (accuracy on the 231 public items, hard tier, calibration);
- the newest Connect Four league runs/*_c4-league*/summary.json (points, TrueSkill, tactical diagnostics), merged into
  the rows by the site on every request (with_connect4), since that summary is rewritten while the league runs.

TrueSkill: every game is a 4-player free-for-all. The winner ranks first; the others are ranked by
the points left in their hands (fewer is better, equal points tie). Games are applied in the order they
finished. The conservative score mu - 3*sigma is the default ranking key.

Written to <run>/leaderboard.json, which the viewer reads (and re-reads while a run is in progress).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import trueskill

from . import stats

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "bench" / "models.json"
RUNS = ROOT / "runs"


def load_registry(path: Path = REGISTRY) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {m["name"]: m for m in json.loads(path.read_text())["models"]}


def trueskill_ratings(games: list[dict]) -> tuple[dict[str, trueskill.Rating], dict[str, list[str]], int]:
    """Ratings after all games, each model's opponents (one entry per game), and how many games were skipped
    because a name occupied several seats (gauntlet copies), which a free-for-all update cannot express."""
    env = trueskill.TrueSkill()
    ratings: dict[str, trueskill.Rating] = defaultdict(env.create_rating)
    opponents: dict[str, list[str]] = defaultdict(list)
    skipped = 0
    for g in games:
        seats = g["seats"]
        if g.get("aborted") or len(set(seats)) != len(seats):
            skipped += 1
            continue
        pts = g["hand_points"]
        order = sorted(set(0 if s == g["winner_seat"] else 1 + pts[s] for s in range(len(seats))))
        ranks = [order.index(0 if s == g["winner_seat"] else 1 + pts[s]) for s in range(len(seats))]
        new = env.rate([(ratings[name],) for name in seats], ranks=ranks)
        for name, (r,) in zip(seats, new):
            ratings[name] = r
        for name in seats:
            opponents[name] += [o for o in seats if o != name]
    return dict(ratings), dict(opponents), skipped


def jevbench_results(runs: Path = RUNS) -> dict[str, dict]:
    """Latest JevBench public-item result per model name found under runs/jevbench-*/<name>/."""
    out: dict[str, dict] = {}
    for summary in sorted(runs.glob("jevbench-*/*/summary.json")):
        name = summary.parent.name
        s = json.loads(summary.read_text())
        rows = [json.loads(line) for line in open(summary.parent / "results.jsonl")]
        hard = [r["correct"] for r in rows if r["task_id"].startswith("hard-")]
        ece = s.get("ece")
        out[name] = {
            "acc": s.get("accuracy"),
            "hard": sum(map(bool, hard)) / len(hard) if hard else None,
            "brier": s.get("brier_mean"),
            "ece": ece.get("ece") if isinstance(ece, dict) else ece,
            "items": s.get("n_attempted"),
            "run": summary.parent.parent.name,
        }
    return out


C4_FIELDS = {  # league summary key -> row field
    "games": "c4_games", "wins": "c4_wins", "draws": "c4_draws", "losses": "c4_losses", "score": "c4_score",
    "ts_mu": "c4_ts_mu", "ts_sigma": "c4_ts_sigma", "ts_score": "c4_ts_score",
    "took_immediate_win": "c4_took_win", "chances_to_win": "c4_win_chances",
    "blocked_immediate_threat": "c4_blocked", "threats_to_block": "c4_threats",
    "p_on_block": "c4_p_block", "p_on_block_chance": "c4_p_block_chance",
    "let_opponent_win_on_top": "c4_gave_win_on_top", "fallbacks": "c4_fallbacks", "avg_latency_ms": "c4_latency_ms",
}


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def connect4_results(runs: Path = RUNS) -> tuple[dict | None, dict[str, dict]]:
    """The newest Connect Four league under runs/*_c4-*/ (config "league": true; two-player matches are ignored):
    its progress, or None without a league, and its results per model name as c4_* row fields."""
    leagues = [d for d in sorted(runs.glob("*_c4-*")) if (_read_json(d / "config.json") or {}).get("league")]
    if not leagues:
        return None, {}
    run = leagues[-1]
    summary = _read_json(run / "summary.json") or {}  # written once the first pair is recorded
    status = summary.get("status") or _read_json(run / "status.json") or {}
    progress = {"run": run.name, "state": status.get("state"), "pairs_done": status.get("pairs_done"),
                "pairs": status.get("pairs"), "games": summary.get("games", status.get("games", 0))}
    results = {name: {field: p.get(key) for key, field in C4_FIELDS.items()}
               for name, p in (summary.get("players") or {}).items()}
    return progress, results


def with_connect4(board: dict, runs: Path = RUNS) -> dict:
    """A copy of the board with the newest Connect Four league merged into its rows (c4_* fields, None for a model
    without games there) and the league's progress as "c4"."""
    progress, results = connect4_results(runs)
    empty = dict.fromkeys(C4_FIELDS.values())
    rows = [{**row, **empty, **results.get(row["name"], {})} for row in board["rows"]]
    return {**board, "rows": rows, "c4": progress}


def build(run_dir: str | Path | None, registry: dict[str, dict] | None = None) -> dict:
    registry = load_registry() if registry is None else registry
    games: list[dict] = []
    summary: dict = {}
    run_meta: dict = {}
    if run_dir is not None:
        run_dir = Path(run_dir)
        with open(run_dir / "games.jsonl") as f:
            for line in f:
                g = json.loads(line)
                games.append({k: g.get(k) for k in ("seats", "winner_seat", "hand_points", "aborted")})
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else stats.summarize(run_dir)
        status_path = run_dir / "status.json"
        run_meta = json.loads(status_path.read_text()) if status_path.exists() else {}
    ratings, opponents, skipped = trueskill_ratings(games)
    return assemble(summary, ratings, opponents, registry, run_dir, len(games), skipped, run_meta)


def assemble(summary: dict, ratings: dict, opponents: dict, registry: dict[str, dict], run_dir, n_games: int,
             skipped: int = 0, run_meta: dict | None = None) -> dict:
    """Leaderboard rows from a summary (stats.summarize / stats.Aggregator), TrueSkill ratings and opponents."""
    run_meta = run_meta or {}
    jb = jevbench_results()
    players = summary.get("players", {})
    # a run with registered models lists every registered model (size shown even before it plays);
    # other runs (heuristic / random baselines) list only their own players
    known = set(players) | set(ratings or {})  # ratings list every player of a live run, before its first game ends
    listed = registry if run_dir is None or any(n in registry for n in known) else {}
    names = list(dict.fromkeys(list(players) + list(listed)))
    rows = []
    for name in names:
        reg = registry.get(name, {})
        p = players.get(name, {})
        r = ratings.get(name)
        opp = opponents.get(name, [])
        mc = p.get("model_calls") or {}
        diag = p.get("diagnostics") or {}
        rate = lambda k: (diag.get(k) or {}).get("rate")  # noqa: E731
        rows.append({
            "name": name,
            "display": reg.get("display", name),
            "repo": reg.get("repo"),
            "caveat": reg.get("caveat"),
            "params": reg.get("params"),
            "params_source": reg.get("params_source"),
            "params_measured": reg.get("params_measured", False),
            "languages": reg.get("languages"),
            "status": reg.get("status"),
            "games": p.get("seat_games", 0),
            "wins": p.get("wins", 0),
            "win_rate": p.get("win_rate") if p else None,
            "win_rate_ci95": p.get("win_rate_ci95"),
            "avg_points_won": p.get("avg_points_won"),
            "avg_points_left": p.get("avg_points_left"),
            "ts_mu": r.mu if r else None,
            "ts_sigma": r.sigma if r else None,
            "ts_score": r.mu - 3 * r.sigma if r else None,
            "avg_opp_mu": sum(ratings[o].mu for o in opp) / len(opp) if opp else None,
            "latency_ms": mc.get("avg_latency_ms"),
            "memory_rounds": mc.get("avg_history_rounds"),
            "max_input_tokens": reg.get("max_input_tokens"),
            "fallback_rate": mc.get("fallback_rate"),
            "drew_with_playable": rate("drew_with_playable"),
            "kept_playable": rate("kept_playable"),
            "wd4_bluff": rate("wd4_bluff"),
            "challenged": rate("challenged"),
            "challenge_right": rate("challenge_right"),
            "jb_acc": (jb.get(name) or {}).get("acc"),
            "jb_hard": (jb.get(name) or {}).get("hard"),
            "jb_brier": (jb.get(name) or {}).get("brier"),
            "jb_ece": (jb.get(name) or {}).get("ece"),
            "jb_items": (jb.get(name) or {}).get("items"),
        })
    return {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run": Path(run_dir).name if run_dir is not None else None,
        "games": n_games,
        "trueskill_skipped_games": skipped,
        "target_games_per_model": run_meta.get("target_games_per_model"),
        "in_progress": run_meta.get("state") == "running",
        "rows": rows,
    }


def write(run_dir: str | Path) -> Path:
    board = build(run_dir)
    path = Path(run_dir) / "leaderboard.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(board, indent=1, ensure_ascii=False))
    tmp.replace(path)  # atomic, so a viewer polling the file never reads half of it
    return path
