"""JevBench public-item results for the bench page.

Per model (runs/jevbench-*/<model>/): accuracy with a 95% interval that resamples whole scenarios, accuracy by
tier (easy / original / hard) and by family, calibration (Brier, ECE), schema validity, paraphrase agreement and
latency. Context: the official JevBench v1.2 per-item outcomes on the same 231 items for every complete system,
so each model can be ranked against them.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .leaderboard import load_registry

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OFFICIAL = ROOT / "third_party" / "jevbench" / "results" / "v1.2" / "jevbench-v1.2-per-task.json"


def _tier(task_id: str) -> str:
    return task_id.split("-", 1)[0]


def model_result(model_dir: Path, tasks: dict[str, dict]) -> dict:
    s = json.loads((model_dir / "summary.json").read_text())
    rows = [json.loads(line) for line in open(model_dir / "results.jsonl")]
    by_tier, by_family, groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        ok = bool(r["correct"])
        by_tier[_tier(r["task_id"])].append(ok)
        by_family[r["family"]].append(ok)
        groups[(tasks.get(r["task_id"]) or {}).get("group") or r["task_id"]].append(ok)
    keys, rng, boots = list(groups), random.Random(0), []
    for _ in range(2000):
        sample = [x for k in (rng.choice(keys) for _ in keys) for x in groups[k]]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    ece = s.get("ece")
    lat = s.get("latency") or {}
    return {
        "acc": s["accuracy"],
        "ci95": [boots[50], boots[1949]],
        "tiers": {t: sum(v) / len(v) for t, v in by_tier.items()},
        "families": {f: sum(v) / len(v) for f, v in sorted(by_family.items())},
        "brier": s.get("brier_mean"),
        "ece": ece.get("ece") if isinstance(ece, dict) else ece,
        "validity": s.get("schema_validity"),
        "paraphrase": (s.get("paraphrase_consistency") or {}).get("agreement"),
        "p50_ms": lat.get("p50_s") * 1000 if lat.get("p50_s") is not None else None,
        "p95_ms": lat.get("p95_s") * 1000 if lat.get("p95_s") is not None else None,
        "attempted": s.get("n_attempted"),
        "failed": (s.get("n_planned") or 0) - (s.get("n_valid") or 0),
        "run": model_dir.parent.name,
    }


def official_context() -> list[dict]:
    if not OFFICIAL.exists():
        return []
    d = json.loads(OFFICIAL.read_text())
    tier = {t["id"]: t["tier"] for t in d["tasks"] if t["public"]}
    out = []
    for key, sysd in d["systems"].items():
        att = {i: o for i, o in sysd["public_tasks"].items() if o[0] != "n"}
        if len(att) < len(tier):
            continue
        hard = [o[0] == "c" for i, o in att.items() if tier[i] == "hard"]
        out.append({"system": sysd["display"], "acc": sum(o[0] == "c" for o in att.values()) / len(att),
                    "hard": sum(hard) / len(hard)})
    return out


def build(runs: Path = RUNS) -> dict:
    registry = load_registry()
    tasks: dict[str, dict] = {}
    models = {}
    for model_dir in sorted(runs.glob("jevbench-*/*/")):
        if not (model_dir / "summary.json").exists():
            continue
        if not tasks and (model_dir.parent / "public_all.jsonl").exists():
            tasks = {json.loads(line)["id"]: json.loads(line) for line in open(model_dir.parent / "public_all.jsonl")}
        name = model_dir.name
        reg = registry.get(name, {})
        models[name] = {"name": name, "display": reg.get("display", name), "repo": reg.get("repo"), "caveat": reg.get("caveat"),
                        "params": reg.get("params"), "params_measured": reg.get("params_measured", False),
                        **model_result(model_dir, tasks)}
    context = official_context()
    everyone = sorted([c["acc"] for c in context] + [m["acc"] for m in models.values()], reverse=True)
    for m in models.values():
        m["overall_rank"] = everyone.index(m["acc"]) + 1
    return {"models": list(models.values()), "official": context, "field_size": len(everyone),
            "items": {"easy": 48, "original": 72, "hard": 111, "total": 231}}
