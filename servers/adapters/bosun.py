"""Bosun v3.1 0.6B (Hanno-Labs) through the authors' Transformers remote code.

Loading and readout are the model card's: AutoModelForCausalLM.from_pretrained("Hanno-Labs/bosun-v3.1-0.6b"
@1d8b6f9, trust_remote_code=True, dtype="auto", device_map="auto") -> modeling_bosun.BosunForDecision (pinned base
Qwen/Qwen3-0.6B@c1899de + PEFT LoRA adapter, not merged + learned decision-token rows), then
`model.predict(state=, instructions=, candidates=, decision_type=)`, which renders the stable-slot JSON prompt,
reads the 256 decision-token logits and returns probabilities in candidate order. Defaults kept: seed 0, row_id "0"
(the candidate presentation shuffle is a fixed function of seed, row_id and the number of candidates).

Candidates follow the card's examples: choice {id: key, label: key, description}, score {id: "i", label: level},
noul {id: "yes", label: "Yes"}, {id: "no", label: "No"} (true/false criteria become their descriptions).
2..255 candidates. serve.py hands structured states over as json.dumps(state); they are decoded back so the
prompt compiler serializes them as it would the original object.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

NAME = "bosun-0.6b"
SNAPSHOT = str(_CACHE / "models--Hanno-Labs--bosun-v3.1-0.6b/snapshots"
             / "1d8b6f9611f9b64b514ce8b57cd86398fbc31a3b")
MAX_INPUT_TOKENS = 32000      # raw request text; Qwen3-0.6B takes 40,960 positions, the prompt adds ~10 tokens/option

_model = None


def _state(s: str):
    """Undo serve.py's json.dumps for dict/list states (exact round trip only)."""
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        return s
    return v if isinstance(v, (dict, list)) and json.dumps(v, ensure_ascii=False) == s else s


def load(device: str) -> list:
    global _model
    from transformers import AutoModelForCausalLM
    _model = AutoModelForCausalLM.from_pretrained(SNAPSHOT, trust_remote_code=True, dtype="auto",
                                                  device_map="auto" if device == "cuda" else device)
    return [_model]


def count_tokens(text: str) -> int:
    return len(_model.tokenizer.encode(text, add_special_tokens=False))


def _predict(state, instructions, candidates, kind) -> list[float]:
    out = _model.predict(state=_state(state), instructions=instructions if instructions is not None else "",
                         candidates=candidates, decision_type=kind)
    return out["probabilities"]


def choice(state, instructions, options):
    keys = list(options)
    p = _predict(state, instructions, [{"id": k, "label": k, "description": options[k]} for k in keys], "choice")
    return dict(zip(keys, p))


def noul(state, instructions, criteria):
    crit = criteria if isinstance(criteria, dict) else {}
    yes = {"id": "yes", "label": "Yes"}
    no = {"id": "no", "label": "No"}
    if crit.get("true"):
        yes["description"] = crit["true"]
    if crit.get("false"):
        no["description"] = crit["false"]
    return _predict(state, instructions, [yes, no], "noul")[0]


def score(state, instructions, levels):
    return _predict(state, instructions, [{"id": str(i), "label": lv} for i, lv in enumerate(levels)], "score")
