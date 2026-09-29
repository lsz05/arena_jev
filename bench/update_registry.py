"""Merge deployment reports into bench/models.json.

    python bench/update_registry.py report.json [report2.json ...]

A report is the JSON array the deployment agents return (see bench/AGENT_BRIEF.md). For each model: status,
measured parameter count (params_measured = true), tokenizer, max_input_tokens, max_options and notes. Values
missing from a report fall back to its servers/models/<name>.toml.
"""

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "bench" / "models.json"

reg = json.loads(REGISTRY.read_text())
by_name = {m["name"]: m for m in reg["models"]}
for path in sys.argv[1:]:
    for r in json.loads(Path(path).read_text()):
        m = by_name.get(r["name"])
        if m is None:
            print(f"unknown model {r['name']}, skipped")
            continue
        toml = ROOT / "servers" / "models" / f"{r['name']}.toml"
        spec = tomllib.loads(toml.read_text()) if toml.exists() else {}
        m["status"] = r.get("status")
        if r.get("status") != "ok":
            m["status_reason"] = r.get("reason")
        if r.get("params"):
            m.update(params=int(r["params"]), params_measured=True,
                     params_source=r.get("params_source") or "counted in the serving process after load")
        for key in ("tokenizer", "max_input_tokens", "max_options"):
            value = r.get(key) if r.get(key) is not None else spec.get(key)
            if value is not None:
                m[key] = value
        if r.get("notes"):
            m["deploy_notes"] = r["notes"]
        print(f"{m['name']:28s} {m.get('status')}  {m['params'] / 1e6:8.2f}M  limit {m.get('max_input_tokens')}  tok {m.get('tokenizer')}")
REGISTRY.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
