"""Ask every model every Connect Four puzzle, each in three option orders, and score the answers.

    python -m arena c4-puzzles-run [--models a,b] [--orders 3] [--per-model 4] [--resume runs/<time>_c4-puzzles]

The options of a puzzle are shuffled once (seeded by its id) and then rotated by a third of their number for each
order, so the correct column sits at a different position each time; averaging over the orders cancels a model's
preference for a position. All options go in one request (they are short); a server that refuses them gets the usual
knockout over its option cap. Output: runs/<time>_c4-puzzles/<model>/results.jsonl, one line per puzzle and order,
and summary.json with the scores (see score()).
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from ..swiss import _answers, model_specs
from ..systemone import SystemOneError
from . import render
from .engine import Connect4Game
from .match import make_player
from .players import HeuristicPlayer

ROOT = Path(__file__).resolve().parents[2]
PUZZLES = ROOT / "data" / "c4-puzzles"
TYPES = ("win", "block", "fork", "solver")


def load_puzzles(root: Path = PUZZLES) -> list[dict]:
    return [json.loads(line) for t in TYPES if (root / f"{t}.jsonl").exists() for line in open(root / f"{t}.jsonl")]


def orders(item: dict, n_orders: int = 3) -> list[list[str]]:
    base = list(item["labels"])
    random.Random(item["id"]).shuffle(base)
    n = len(base)
    return [base[k * n // n_orders:] + base[:k * n // n_orders] for k in range(n_orders)]


def ask(full, capped, item: dict, names: list[str], rng: random.Random) -> dict:
    game = Connect4Game([int(c) - 1 for c in item["moves"]])
    decision = game.decision()
    crit = item["question"]["criteria"]
    q = render.Question("move", item["question"]["instructions"], {n: crit[n] for n in names})
    info: dict = {"calls": []}
    knockout = False
    try:
        pick = full._ask(game, decision, q, rng, info)
    except SystemOneError as e:
        if len(names) <= capped.max_options or "HTTP 4" not in str(e):
            raise
        info, knockout = {"calls": []}, True
        pick = capped._ask(game, decision, q, rng, info)
    calls = info["calls"]
    probs = calls[0]["probabilities"] if len(calls) == 1 else None
    return {"pick": pick, "correct": pick == item["expected"], "probs": probs,
            "p_correct": probs.get(item["expected"]) if probs else None, "calls": len(calls), "knockout": knockout,
            "with_moves": calls[0].get("with_moves") if calls else None,
            "latency_ms": round(sum(c["latency_ms"] for c in calls), 1)}


def run_model(name: str, puzzles: list[dict], out: Path, n_orders: int, per_model: int, progress: dict) -> None:
    full = make_player(name)
    capped = make_player(name)
    full.max_options = 255  # try every option in one request first
    path = out / name / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {(r["item"], r["order"]) for r in map(json.loads, open(path)) if r.get("error") is None}
    tasks = [(it, k, names) for it in puzzles for k, names in enumerate(orders(it, n_orders)) if (it["id"], k) not in done]
    progress[name] = [len(done), len(done) + len(tasks)]
    lock = threading.Lock()

    def one(task):
        it, k, names = task
        row = {"item": it["id"], "family": it["family"], "order": k, "options": names}
        try:
            row.update(ask(full, capped, it, names, random.Random(f"{it['id']}:{k}")))
            row["error"] = None
        except (SystemOneError, KeyError, ValueError) as e:
            row["error"] = f"{type(e).__name__}: {e}"
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress[name][0] += 1

    with ThreadPoolExecutor(max_workers=per_model) as pool:
        list(pool.map(one, tasks))


def run(models: list[str] | None = None, exclude: list[str] | None = None, n_orders: int = 3, per_model: int = 4,
        out_root: str | Path = "runs", resume: str | Path | None = None) -> Path:
    puzzles = load_puzzles()
    if not puzzles:
        raise SystemExit(f"no puzzles in {PUZZLES}: run `python -m arena c4-puzzles-make` first")
    names = [s["name"] for s in model_specs({"models": models or [], "exclude": exclude or []}) if _answers(s)]
    if resume:
        out = Path(resume)
    else:
        out = Path(out_root) / f"{datetime.now():%Y%m%d-%H%M%S}_c4-puzzles"
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps({"puzzles": len(puzzles), "orders": n_orders, "models": names,
                                                     "per_model": per_model}, indent=1))
    print(f"{len(names)} models x {len(puzzles)} puzzles x {n_orders} orders -> {out}", flush=True)
    progress: dict[str, list[int]] = {}
    threads = [threading.Thread(target=run_model, args=(n, puzzles, out, n_orders, per_model, progress), daemon=True)
               for n in names]
    for t in threads:
        t.start()
    last = 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        if time.time() - last > 120:
            last = time.time()
            done = sum(p[0] for p in progress.values())
            total = sum(p[1] for p in progress.values()) or 1
            slow = min(progress.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]), default=None)
            print(f"[{datetime.now():%H:%M:%S}] {done}/{total} answers" + (f"; slowest {slow[0]} {slow[1][0]}/{slow[1][1]}" if slow else ""),
                  flush=True)
            write_summary(out, puzzles)
    write_summary(out, puzzles, final=True)
    print(json.dumps({n: round(s["accuracy"], 3) for n, s in json.loads((out / "summary.json").read_text())["models"].items()},
                     indent=1), flush=True)
    return out


# ---- scores ------------------------------------------------------------------------------------------


def score(rows: list[dict], puzzles: dict[str, dict]) -> dict:
    """Scores of one model from its result rows (one per puzzle and order)."""
    ok = [r for r in rows if r.get("error") is None]
    by_item = defaultdict(dict)
    for r in ok:
        by_item[r["item"]][r["order"]] = r
    mean = lambda v: sum(v) / len(v) if v else None  # noqa: E731
    with_p = [r for r in ok if r.get("probs")]
    brier = [sum((p - (k == puzzles[r["item"]]["expected"])) ** 2 for k, p in r["probs"].items()) for r in with_p]
    bins = defaultdict(list)  # calibration of the top answer
    for r in with_p:
        top = max(r["probs"].values())
        bins[min(9, int(top * 10))].append((top, r["correct"]))
    ece = sum(len(b) * abs(mean([t for t, _ in b]) - mean([c for _, c in b])) for b in bins.values()) / len(with_p) if with_p else None
    avg_correct = []
    for item, per in by_item.items():
        if len(per) >= 2 and all(r.get("probs") for r in per.values()):
            labels = puzzles[item]["labels"]
            avg = {k: mean([r["probs"][k] for r in per.values()]) for k in labels}
            avg_correct.append(max(avg, key=avg.get) == puzzles[item]["expected"])
    consistent = [len({r["pick"] for r in per.values()}) == 1 for per in by_item.values() if len(per) > 1]
    pos = Counter(r["options"].index(r["pick"]) for r in ok if len(r["options"]) == 7)
    cols = Counter(r["pick"] for r in ok if len(r["options"]) == 7)
    n7 = sum(pos.values())
    fam = {}
    for t in TYPES:
        rs = [r for r in ok if r["family"] == t]
        fam[t] = {"accuracy": mean([r["correct"] for r in rs]),
                  "p_correct": mean([r["p_correct"] for r in rs if r.get("p_correct") is not None])}
    return {
        "answers": len(ok), "errors": len(rows) - len(ok), "items": len(by_item),
        "accuracy": mean([r["correct"] for r in ok]),
        "accuracy_first_order": mean([r["correct"] for r in ok if r["order"] == 0]),
        "accuracy_avg_probs": mean(avg_correct),
        "p_correct": mean([r["p_correct"] for r in with_p]),
        "brier": mean(brier), "ece": ece,
        "consistency": mean(consistent),
        "position_first": pos[0] / n7 if n7 else None,
        "position_max": max(pos.values()) / n7 if n7 else None,
        "column_max": max(cols.values()) / n7 if n7 else None, "column_favourite": cols.most_common(1)[0][0] if cols else None,
        "knockout_share": mean([r["knockout"] for r in ok]),
        "latency_ms": mean([r["latency_ms"] for r in ok]),
        "by_type": fam,
    }


def baselines(puzzles: list[dict]) -> dict:
    """Random guessing, always the most central column, and the rule-based heuristic (wins, blocks, centre)."""
    out = {}
    chance = lambda its: sum(1 / len(i["labels"]) for i in its) / len(its) if its else None  # noqa: E731
    central = lambda i: min(i["labels"], key=lambda c: (abs(int(c) - 4), int(c)))  # noqa: E731
    h = HeuristicPlayer()

    def heuristic(i):
        g = Connect4Game([int(c) - 1 for c in i["moves"]])
        return str(h.act(g, g.decision(), random.Random(i["id"])).col + 1)

    for name, fn in (("random", None), ("centre", central), ("heuristic", heuristic)):
        row = {"accuracy": chance(puzzles) if fn is None else sum(fn(i) == i["expected"] for i in puzzles) / len(puzzles),
               "by_type": {}}
        for t in TYPES:
            its = [i for i in puzzles if i["family"] == t]
            row["by_type"][t] = {"accuracy": chance(its) if fn is None else
                                 (sum(fn(i) == i["expected"] for i in its) / len(its) if its else None)}
        out[name] = row
    return out


def write_summary(out: Path, puzzles: list[dict], final: bool = False) -> None:
    index = {p["id"]: p for p in puzzles}
    models = {}
    for d in sorted(out.iterdir()):
        path = d / "results.jsonl"
        if d.is_dir() and path.exists():
            rows = [json.loads(line) for line in open(path) if line.endswith("\n")]
            if rows:
                models[d.name] = score(rows, index)
    summary = {"state": "done" if final else "running", "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "puzzles": len(puzzles), "by_type": dict(Counter(p["family"] for p in puzzles)),
               "orders": json.loads((out / "config.json").read_text()).get("orders", 3) if (out / "config.json").exists() else 3,
               "models": models, "baselines": baselines(puzzles)}
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, out / "summary.json")
