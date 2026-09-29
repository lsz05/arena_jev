"""Adapter for lostargon/Tiny-Jev (servers/adapters/serve.py).

Loads the model exactly as the model card does (AutoModel + trust_remote_code -> modeling_tiny_jev.TinyJevModel,
pinned revision) and answers through its own typed API: model.choice / model.noul / model.score. Choice options are
rendered by the authors' code as "key: description" lines; noul is the authors' two-option ["no", "yes"] choice over
the instructions only (their API takes no criteria, so noul criteria are not shown to the model); score is a choice
over the ordered levels.

Context: the authors' encode() fits everything into 4096 tokens per question and, when the state does not fit,
silently keeps its head and tail. This adapter measures the full prompt with the authors' encode() and refuses
(HTTP 422 "maximum context length") instead of letting it truncate. No option cap (the card tests 151).
"""

from __future__ import annotations

REPO = "lostargon/Tiny-Jev"
REVISION = "62449fb3272d68fb8c8fe8a33082566643a76d0b"

NAME = "tiny-jev-0.6b"
MAX_LEN = 4096                 # TinyJevModel.encode(max_len=4096), the authors' per-question context
MAX_INPUT_TOKENS = MAX_LEN     # coarse pre-check in serve.py; the exact check (with the prompt wrapper) is below

_model = None
_tok = None


def load(device: str) -> list:
    global _model, _tok
    from transformers import AutoModel, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    _model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).to(device).eval()
    return [_model]


def count_tokens(text: str) -> int:
    return len(_tok(text, add_special_tokens=False)["input_ids"])


def _check(state: str, question: str, options: list[str]) -> None:
    ids, _ = _model.encode(_tok, state, question, options, max_len=10**9)   # untruncated length, authors' layout
    if len(ids) > MAX_LEN:
        raise ValueError(f"maximum context length is {MAX_LEN} tokens (Tiny-Jev prompt), the prompt has {len(ids)}")


def choice(state, instructions, options):
    _check(state, instructions or "", [f"{k}: {v}" for k, v in options.items()])
    return _model.choice(_tok, state, instructions or "", dict(options))["probabilities"]


def noul(state, instructions, criteria):
    _check(state, instructions or "", ["no", "yes"])
    return _model.noul(_tok, state, instructions or "")


def score(state, instructions, levels):
    _check(state, instructions or "", list(levels))
    # model.score() == decide() == probabilities() over the levels; call the latter so duplicate level texts
    # keep one probability each (score() returns them keyed by text)
    return _model.probabilities(_tok, state, [(instructions or "", list(levels))])[0]
