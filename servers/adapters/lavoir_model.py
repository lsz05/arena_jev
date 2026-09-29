"""Adapter for moganai/lavoir (ModernBERT-large decision model with a VOI head; weights CC-BY-NC-4.0, research use).

Code path: the authors' `lavoir` package (git clone github.com/moganai/lavoir @ af79b0a7a7f9, servers/src/lavoir,
pip -e into venvs/c-laya, with laya 0.3.21 from pip) on the HF checkpoint servers/src/lavoir-weights (snapshot of
moganai/lavoir @ 4c5eaeb99b23). Lavoir.from_pretrained(dir, device) and Lavoir.predict(state, question) without
slots: the card's "plain calibrated classifier" use (Laya's input format, general-data temperatures). CUDA runs
the package's own bf16 autocast. ModernBERT's reference_compile is switched off (eager, as laya.Agent defaults to).

Limits (checkpoint config + lavoir.items.make_item): max_len 1024 tokens; question + options at most 256 tokens
when no slots are given; each option at most 48 tokens. The package would silently cut to fit; laya_guard refuses
instead (HTTP 422, nothing truncated).
"""

from __future__ import annotations

from pathlib import Path

import laya_guard

NAME = "lavoir"
CKPT = Path(__file__).resolve().parent.parent / "src" / "lavoir-weights"
MAX_INPUT_TOKENS = 1024  # pre-check on the joined text; the exact check is laya_guard on the real sequence
HEAD_MAX_LEN_NOSLOT = 256  # lavoir.items.make_item default for slot-free requests

_m = None


def load(device: str) -> list:
    global _m
    from lavoir import Lavoir

    _m = Lavoir.from_pretrained(str(CKPT), device=device)
    # ModernBERT's reference_compile="auto" torch.compiles the embedding block on CUDA. Keep the eager path, as
    # Laya's own Agent does by default: same weights and math, and the compile's weakrefs crash the parameter probe.
    _m.model.encoder.config.reference_compile = False
    return [_m.model]


def count_tokens(text: str) -> int:
    return len(_m.tokenizer(text, add_special_tokens=False)["input_ids"])


def _predict(state, question: dict) -> dict:
    from lavoir.items import to_internal

    laya_guard.check(_m.tokenizer, state, to_internal(question), _m.max_len, HEAD_MAX_LEN_NOSLOT)
    return _m.predict(state, question).probabilities


def choice(state, instructions, options: dict) -> dict:
    if len(options) == 1:
        return {k: 1.0 for k in options}
    p = _predict(state, {"type": "choice", "instructions": instructions, "criteria": dict(options)})
    return {k: float(p[k]) for k in options}


def noul(state, instructions, criteria) -> float:
    q = {"type": "noul", "instructions": instructions}
    if isinstance(criteria, dict) and criteria:
        q["criteria"] = {k: v for k, v in criteria.items() if k in ("true", "false")}
    p = _predict(state, q)
    return float(p["true"])


def score(state, instructions, levels: list) -> list:
    p = _predict(state, {"type": "score", "instructions": instructions, "criteria": list(levels)})
    return [float(p[str(i)]) for i in range(len(levels))]
