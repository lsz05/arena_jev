"""Adapter for SupersonicLabs/Julia-1 (144.3M, mmBERT-small + decision head).

Code path: the authors' `julia` package from the HF repo (servers/src/julia-1 = snapshot at a85b127321d5, installed
editable into venvs/c-julia with transformers 5.0.0 as pinned by its pyproject). Loaded exactly as the model card's
"Start here" snippet: load_model(dir, device, strict_encoding=True, max_length=8192, head_length=512), then the
named-question API engine.predict(state=..., questions=...) (full softmax over the marker logits). On CUDA the
package runs BF16 autocast (its own default; the card's reported numbers are H200 BF16).

Limits (the package's strict encoding, nothing truncated): 2-20 options (choice and score levels), each option
<= 48 tokens, question + options <= 512 tokens (head budget), whole packed sequence <= 8192 tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path

NAME = "julia-1"
CKPT = Path(__file__).resolve().parent.parent / "src" / "julia-1"
MAX_INPUT_TOKENS = 8192  # pre-check on the joined request text; the exact check is the package's strict encoding
MAX_OPTIONS = 20

_engine = None


def _too_long(msg: str) -> Exception:
    cls = getattr(sys.modules.get("__main__"), "TooLong", ValueError)
    return cls(msg)


def load(device: str) -> list:
    global _engine
    from julia import load_model

    _engine = load_model(str(CKPT), device=device, strict_encoding=True, max_length=8192, head_length=512)
    return [_engine.model]


def count_tokens(text: str) -> int:
    return len(_engine.tokenizer(text, add_special_tokens=False)["input_ids"])


def _ask(state, question: dict) -> dict:
    try:
        out = _engine.predict(state=state, questions={"decision": question})
    except ValueError as e:
        msg = str(e)
        if "exceed" in msg or "budget" in msg:
            raise _too_long(f"maximum context length is 8192 tokens (question+options head 512, option 48): {msg}") from e
        raise
    return out["answers"]["decision"]


def choice(state, instructions, options: dict) -> dict:
    if len(options) == 1:  # the package needs >= 2 options; a single option is the only answer
        return {k: 1.0 for k in options}
    if len(options) > MAX_OPTIONS:
        raise ValueError(f"julia-1 accepts at most {MAX_OPTIONS} options per call (native limit), got {len(options)}")
    ans = _ask(state, {"type": "choice", "instructions": instructions, "criteria": dict(options)})
    return {k: float(ans["probabilities"][k]) for k in options}


def noul(state, instructions, criteria) -> float:
    crit = None
    if isinstance(criteria, dict) and criteria.get("true") and criteria.get("false"):
        crit = {"false": criteria["false"], "true": criteria["true"]}
    q = {"type": "noul", "instructions": instructions}
    if crit is not None:
        q["criteria"] = crit
    return float(_ask(state, q)["noul"])


def score(state, instructions, levels: list) -> list:
    ans = _ask(state, {"type": "score", "instructions": instructions, "criteria": list(levels)})
    return [float(ans["probabilities"][str(i)]) for i in range(len(levels))]
