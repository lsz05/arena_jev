"""Decision-1.0-Eos-0.8B (llm-semantic-router) through the authors' own local inference package.

The current Hub revision (363c4a5) is "model-only" and points to a vLLM Semantic Router Decision runtime that
has not been released. The previous revision 3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd ships the authors' native
package `decision/` (DecisionModel.from_pretrained / decide) next to byte-identical weights (same LFS sha256 for
backbone/model.safetensors, decision_head.safetensors, tokenizer), so this adapter loads that snapshot and calls
`DecisionModel.decide(state, {name: question})` unchanged: BF16 backbone, FP32 head, T = 1.0389139156246665,
16,384-token complete-input limit (the package rejects longer inputs; no truncation).

serve.py hands structured states over as json.dumps(state); they are decoded back so the package renders them
exactly as it would the original object (its own canonical JSON).
"""

from __future__ import annotations

import json
import sys

NAME = "decision-eos-0.8b"
SNAPSHOT = ("/home/intuser/.cache/huggingface/hub/models--llm-semantic-router--Decision-1.0-Eos-0.8B/snapshots/"
            "3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd")
LIMIT = 16384                 # the package's max complete input (state + question + candidates + formatting)
MAX_INPUT_TOKENS = 16300      # pre-check on the raw request text; the package's exact check runs after it

_engine = None


def _too_long(msg: str) -> Exception:
    cls = getattr(sys.modules.get("__main__"), "TooLong", ValueError)
    return cls(msg)


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
    from decision import DecisionModel  # the authors' package from the snapshot
    _engine = DecisionModel.from_pretrained(SNAPSHOT, device="cuda:0" if device == "cuda" else device,
                                            max_length=LIMIT)
    return [_engine.model]


def count_tokens(text: str) -> int:
    return len(_engine.tokenizer.encode(text, add_special_tokens=False))


def _ask(state: str, question: dict) -> dict:
    try:
        out = _engine.decide(_state(state), {"q": question})
    except ValueError as e:
        if "exceeds max_length" in str(e):
            raise _too_long(f"maximum context length is {LIMIT} tokens: {e}") from e
        raise
    return out["answers"]["q"]


def choice(state, instructions, options):
    return _ask(state, {"type": "choice", "instructions": instructions, "criteria": dict(options)})["probabilities"]


def noul(state, instructions, criteria):
    q = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    return _ask(state, q)["noul"]


def score(state, instructions, levels):
    p = _ask(state, {"type": "score", "instructions": instructions, "criteria": list(levels)})["probabilities"]
    return [p[str(i)] for i in range(len(levels))]
