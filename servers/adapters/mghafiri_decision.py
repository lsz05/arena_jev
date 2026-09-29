"""Adapter for mghafiri/qwen3.5-0.8B-decision-model (servers/adapters/serve.py).

Uses the authors' own `jevlite` package bundled in the HF repo (jevlite.model.SystemOne, the model-card usage):
plain-text prompt "### State / ### Question / ### Options ... / Answer:", softmax over the single-token labels
" A".." Z"," a".." z" (choice, score) or " yes"/" no" (noul), with the per-type temperatures of calibration.json.
Nothing is re-implemented: every question goes through SystemOne.distributions().

Limits (the authors'): at most 52 options (A-Z, a-z; jevlite raises beyond that -> HTTP 422); a prompt budget of
3072 tokens (jevlite's DEFAULT_MAX_TOKENS). jevlite would silently drop the middle of an over-long prompt; this
adapter refuses such requests instead (HTTP 422 "maximum context length"), so nothing is truncated.
"""

from __future__ import annotations

import sys

REPO = "mghafiri/qwen3.5-0.8B-decision-model"
REVISION = "4a9939034a27006b4b62ad5aed65a90b82d0d350"

NAME = "qwen3.5-0.8b-decision"
PROMPT_BUDGET = 3072          # jevlite.model.DEFAULT_MAX_TOKENS
# serve.py checks the request text (state, instructions, "key: value" lines) against this before the model runs;
# jevlite's own headers and "A. " labels add ~20 + 2 per option tokens on top (<= ~125 at 52 options).
MAX_INPUT_TOKENS = 2940

_engine = None


def load(device: str) -> list:
    global _engine
    from huggingface_hub import snapshot_download
    path = snapshot_download(REPO, revision=REVISION)
    sys.path.insert(0, path)                     # the bundled jevlite/ package, as on the model card
    from jevlite.model import SystemOne
    _engine = SystemOne(path, device=device)     # bf16 default, applies calibration.json
    return [_engine.model]


def count_tokens(text: str) -> int:
    return len(_engine.tokenizer.encode(text, add_special_tokens=False))


def _dist(state: str, question: dict) -> list[float]:
    from jevlite.prompt import render
    n = len(_engine.tokenizer.encode(render(state, question), add_special_tokens=False))
    if n > PROMPT_BUDGET:
        raise ValueError(f"maximum context length is {PROMPT_BUDGET} tokens (jevlite prompt budget), "
                         f"the rendered prompt has {n}")
    (p,), _ = _engine.distributions([(state, question, None)])
    return p


def choice(state, instructions, options):
    q = {"type": "choice", "instructions": instructions or "", "criteria": dict(options)}
    return dict(zip(options, _dist(state, q)))


def noul(state, instructions, criteria):
    q = {"type": "noul", "instructions": instructions or "", "criteria": criteria}
    return _dist(state, q)[0]                     # P(" yes")


def score(state, instructions, levels):
    q = {"type": "score", "instructions": instructions or "", "criteria": list(levels)}
    return _dist(state, q)
