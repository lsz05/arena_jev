"""Intern-Decision-0.8B (internlm) through the model card's own inference module.

`inference.py` shipped in HF internlm/Intern-Decision-0.8B@85a0cc5 (DecisionEngine, backend "hf", the card's
Quick start) is imported from the snapshot and called unchanged, text only: one `engine.predict({"state",
"questions": {name: question}})` per question. Defaults kept: bf16, sdpa, max_length 8192 (longer inputs are
rejected by the module, no truncation), checkpoint temperature 2.747760550703, up to 62 options per question
(single-token symbols A-Z a-z 0-9). Native choice / score / noul.

serve.py hands structured states over as json.dumps(state); they are decoded back so the module renders them
exactly as it would the original object (its own indented JSON).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

NAME = "intern-decision-0.8b"
SNAPSHOT = str(_CACHE / "models--internlm--Intern-Decision-0.8B/snapshots"
             / "85a0cc5a99d67ea8d56dfe98115689212867171d")
LIMIT = 8192                  # DecisionEngine(max_length=8192): whole rendered prompt incl. system prompt/schema
MAX_INPUT_TOKENS = 8000       # pre-check on the raw request text; the module's exact check runs after it

_engine = None


def _state(s: str):
    """Undo serve.py's json.dumps for dict/list states (exact round trip only)."""
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        return s
    return v if isinstance(v, (dict, list)) and json.dumps(v, ensure_ascii=False) == s else s


def load(device: str) -> list:
    global _engine
    sys.path.insert(0, SNAPSHOT)
    from inference import DecisionEngine  # the model card's module
    _engine = DecisionEngine(checkpoint=SNAPSHOT, device=device, max_length=LIMIT)
    return [_engine.backend.model]


def count_tokens(text: str) -> int:
    return len(_engine.tokenizer.encode(text, add_special_tokens=False))


def _ask(state: str, question: dict) -> dict:
    try:
        out = _engine.predict({"state": _state(state), "questions": {"q": question}})
    except ValueError as e:
        if "truncation is forbidden" in str(e):
            cls = getattr(sys.modules.get("__main__"), "TooLong", ValueError)
            raise cls(f"maximum context length is {LIMIT} tokens: {e}") from e
        raise
    return out["answers"]["q"]


def choice(state, instructions, options):
    q = {"type": "choice", "criteria": dict(options)}
    if instructions is not None:
        q["instructions"] = instructions
    return _ask(state, q)["probabilities"]


def noul(state, instructions, criteria):
    q = {"type": "noul"}
    if instructions is not None:
        q["instructions"] = instructions
    if criteria:
        q["criteria"] = criteria
    return _ask(state, q)["noul"]


def score(state, instructions, levels):
    q = {"type": "score", "criteria": list(levels)}
    if instructions is not None:
        q["instructions"] = instructions
    p = _ask(state, q)["probabilities"]
    return [p[str(i)] for i in range(len(levels))]
