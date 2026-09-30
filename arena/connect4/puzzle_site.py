"""Data for the site's Connect Four puzzles page (/arenas/c4puzzles/): the newest runs/*_c4-puzzles run."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ..leaderboard import load_registry
from .puzzle_run import TYPES, baselines, load_puzzles, orders

RUNS = Path(__file__).resolve().parents[2] / "runs"
_CACHE: dict = {"sig": None, "answers": {}}


def newest_run(runs: Path = RUNS) -> Path | None:
    found = [d for d in sorted(runs.glob("*_c4-puzzles")) if (d / "config.json").exists()]
    return found[-1] if found else None


def answers(run: Path) -> dict[str, dict[str, dict[int, dict]]]:
    """item id -> model -> order -> result row, read again only when a results file changes."""
    files = sorted(run.glob("*/results.jsonl"))
    sig = (str(run), tuple((str(f), f.stat().st_mtime, f.stat().st_size) for f in files))
    if sig != _CACHE["sig"]:
        data: dict = defaultdict(lambda: defaultdict(dict))
        for f in files:
            for line in open(f):
                if line.endswith("\n"):
                    r = json.loads(line)
                    if r.get("error") is None:
                        data[r["item"]][f.parent.name][r["order"]] = r
        _CACHE.update(sig=sig, answers={k: dict(v) for k, v in data.items()})
    return _CACHE["answers"]


def summary(runs: Path = RUNS) -> dict:
    run = newest_run(runs)
    puzzles = load_puzzles()
    if _CACHE.get("baselines") is None:  # needs no model: computed once
        _CACHE["baselines"] = baselines(puzzles)
    out = {"run": run.name if run else None, "puzzles": len(puzzles), "by_type": dict(Counter(p["family"] for p in puzzles)),
           "state": None, "models": [], "baselines": _CACHE["baselines"]}
    if run is None or not (run / "summary.json").exists():
        return out
    s = json.loads((run / "summary.json").read_text())
    registry = load_registry()
    models = []
    for name, m in s["models"].items():
        reg = registry.get(name, {})
        models.append({"name": name, "display": reg.get("display", name), "repo": reg.get("repo"), "caveat": reg.get("caveat"),
                       "params": reg.get("params"), "params_measured": reg.get("params_measured", False),
                       "params_source": reg.get("params_source"),
                       **{k: v for k, v in m.items() if k != "by_type"},
                       **{f"{t}_accuracy": (m["by_type"].get(t) or {}).get("accuracy") for t in TYPES}})
    out.update(state=s.get("state"), updated=s.get("updated"), orders=s.get("orders"), models=models,
               baselines=s.get("baselines", {}))
    return out


def items(runs: Path = RUNS) -> dict:
    run = newest_run(runs)
    got = answers(run) if run else {}
    rows = []
    for p in load_puzzles():
        per = got.get(p["id"], {})
        results = [r for by_order in per.values() for r in by_order.values()]
        ps = [r["p_correct"] for r in results if r.get("p_correct") is not None]
        rows.append({"id": p["id"], "family": p["family"], "tier": p["tier"], "expected": p["expected"], "ply": p["ply"],
                     "mover": p["mover"], "legal": len(p["labels"]), "models": len(per),
                     "right": sum(r["correct"] for r in results), "answers": len(results),
                     "p_correct": round(sum(ps) / len(ps), 3) if ps else None})
    return {"items": rows, "types": list(TYPES)}


def item(item_id: str, runs: Path = RUNS) -> dict:
    puzzle = next((p for p in load_puzzles() if p["id"] == item_id), None)
    if puzzle is None:
        raise KeyError(item_id)
    run = newest_run(runs)
    per = (answers(run) if run else {}).get(item_id, {})
    registry = load_registry()
    models, picks = [], Counter()
    for name, by_order in per.items():
        results = [by_order[k] for k in sorted(by_order)]
        for r in results:
            picks[r["pick"]] += 1
        ps = [r["p_correct"] for r in results if r.get("p_correct") is not None]
        models.append({"name": name, "display": registry.get(name, {}).get("display", name),
                       "orders": [{"order": r["order"], "pick": r["pick"], "correct": r["correct"],
                                   "p_correct": r.get("p_correct"), "position": r["options"].index(r["pick"]) + 1}
                                  for r in results],
                       "p_correct": sum(ps) / len(ps) if ps else None})
    models.sort(key=lambda m: (-(m["p_correct"] if m["p_correct"] is not None else -1), m["display"]))
    return {"puzzle": puzzle, "orders": orders(puzzle), "models": models, "picks": dict(picks),
            "answers": sum(picks.values())}
