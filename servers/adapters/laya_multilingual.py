"""Adapter for convaiinnovations/laya-multilingual (mmBERT-base + Laya decision head, 322M).

Code path: the authors' `laya` package 0.3.21 (pip laya[serve], venvs/c-laya) on the HF checkpoint
servers/src/laya-multilingual (snapshot @ e4e9ddf21a7b). laya.load(dir, device) and Agent.predict(state,
{"decision": question}, max_len=8192): the card's documented setting for long inputs ("It ships with a 1,024-token
limit that cuts long documents off, so pass max_len=8192"). CUDA uses the checkpoint's bf16 autocast (package
default); ModernBERT compile stays off (Agent default).

Why not `laya-serve` as is: it routes each request by script to one of several Laya checkpoints (English text would
go to the English `laya` checkpoint, not this one), and it silently truncates at the checkpoint's 1,024 tokens
unless the client sends max_len. The model computation here is the same Agent.predict it calls.

Limits: 8,192 tokens total; question + options at most 256 tokens (checkpoint head_max_len); each option at most
48 tokens. Laya would cut silently to fit; laya_guard refuses instead (HTTP 422).
"""

from __future__ import annotations

from pathlib import Path

import laya_guard

NAME = "laya-multilingual"
CKPT = Path(__file__).resolve().parent.parent / "src" / "laya-multilingual"
MAX_LEN = 8192
MAX_INPUT_TOKENS = MAX_LEN  # pre-check on the joined text; the exact check is laya_guard on the real sequence

_agent = None


def load(device: str) -> list:
    global _agent
    import laya

    _agent = laya.load(str(CKPT), device=device)
    # laya falls back to CPU silently (a warning) when the GPU placement fails, e.g. out of memory on the shared
    # GPU; refuse to serve in that case instead of answering 10x slower on another device.
    actual = next(_agent.model.parameters()).device
    if device.startswith("cuda") and actual.type != "cuda":
        raise RuntimeError(f"laya placed the model on {actual}, not {device} (see the laya warning above)")
    return [_agent.model]


def count_tokens(text: str) -> int:
    return len(_agent.tok(text, add_special_tokens=False)["input_ids"])


def _ask(state, question: dict) -> dict:
    laya_guard.check(_agent.tok, state, _agent._to_internal(question), MAX_LEN, _agent.cfg.get("head_max_len", 192))
    out = _agent.predict(state, {"decision": question}, max_len=MAX_LEN)
    return out["answers"]["decision"]


def choice(state, instructions, options: dict) -> dict:
    if len(options) == 1:
        return {k: 1.0 for k in options}
    ans = _ask(state, {"type": "choice", "instructions": instructions, "criteria": dict(options)})
    return {k: float(ans["probabilities"][k]) for k in options}


def noul(state, instructions, criteria) -> float:
    q = {"type": "noul", "instructions": instructions}
    if isinstance(criteria, dict) and criteria:
        q["criteria"] = {k: v for k, v in criteria.items() if k in ("true", "false")}
    return float(_ask(state, q)["noul"])


def score(state, instructions, levels: list) -> list:
    ans = _ask(state, {"type": "score", "instructions": instructions, "criteria": list(levels)})
    return [float(ans["probabilities"][str(i)]) for i in range(len(levels))]
