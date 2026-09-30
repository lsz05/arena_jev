"""Tev1-0.8B-experimental (togethercomputer) with the authors' decision contract.

Request construction is the authors' own code: `payload()` from togethercomputer/tev1 examples/decide.py (github,
commit 1dde778), i.e. their system instruction plus json.dumps({"state", "question", "options": [{label A..X, key,
description}]}) as the user turn, chat template with enable_thinking=False. The authors decode with temperature 0,
max_tokens 8 and a regex constraint to the listed letters; the constrained first token is the answer. Here the
same constrained step is read directly: the next-token logits after the rendered prompt, softmax over the allowed
option letters only (argmax == the constrained greedy answer). 2..24 options (the contract's hard cap, A..X).

noul follows the authors' training data (build_dataset.py, boolq -> kind "noul"): options yes: "Yes." / no: "No."
(descriptions from the request's true/false criteria when given); P(yes) is returned. score uses serve.py's
fallback, which is the training convention for score (keys "0".."L-1", levels in canonical order).
Checkpoint: HF togethercomputer/Tev1-0.8B-experimental@6bb2dff, loaded whole (Qwen3_5ForConditionalGeneration,
bf16, vision tower included as published).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_CACHE = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))

NAME = "tev1-0.8b"
SNAPSHOT = str(_CACHE / "models--togethercomputer--Tev1-0.8B-experimental/snapshots"
             / "6bb2dff14b38fea90ddb14d870166ccaf77374e9")
REPO = str(Path(__file__).resolve().parent.parent / "src" / "tev1")
MAX_INPUT_TOKENS = 32000
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWX"

_tok = _model = _payload = _torch = _device = None
_label_ids: list[int] = []


def _state(s: str):
    """Undo serve.py's json.dumps for dict/list states (exact round trip only), so they render as the authors would."""
    try:
        v = json.loads(s)
    except (ValueError, TypeError):
        return s
    return v if isinstance(v, (dict, list)) and json.dumps(v, ensure_ascii=False) == s else s


def load(device: str) -> list:
    global _tok, _model, _payload, _torch, _device, _label_ids
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    sys.path.insert(0, REPO)
    from examples.decide import payload  # the authors' request builder
    _torch, _payload, _device = torch, payload, device
    _tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    _model = AutoModelForImageTextToText.from_pretrained(SNAPSHOT, dtype=torch.bfloat16).to(device).eval()
    # Each letter must be one token right after the generation prompt (checked in context).
    probe = _render({"state": "x", "question": "q", "options": [
        {"label": "A", "key": "a", "description": "a"}, {"label": "B", "key": "b", "description": "b"}]})
    base = _tok.encode(probe, add_special_tokens=False)
    for c in LABELS:
        ids = _tok.encode(probe + c, add_special_tokens=False)
        if ids[:len(base)] != base or len(ids) != len(base) + 1:
            raise RuntimeError(f"label {c!r} is not a single token after the prompt")
        _label_ids.append(ids[-1])
    return [_model]


def _render(record: dict) -> str:
    body = _payload(record, NAME)
    return _tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True,
                                    **body["chat_template_kwargs"])


def count_tokens(text: str) -> int:
    return len(_tok.encode(text, add_special_tokens=False))


def _decide(state: str, question: str | None, options: list[tuple[str, str]]) -> dict[str, float]:
    if not 2 <= len(options) <= len(LABELS):
        raise ValueError(f"Tev1 takes 2-{len(LABELS)} options, got {len(options)}")
    record = {"state": _state(state), "question": question or "",
              "options": [{"label": LABELS[i], "key": k, "description": d} for i, (k, d) in enumerate(options)]}
    text = _render(record)  # payload() validates: nonempty question/keys/descriptions, unique keys
    ids = _tok.encode(text, add_special_tokens=False)
    if len(ids) > MAX_INPUT_TOKENS + 1024:
        cls = getattr(sys.modules.get("__main__"), "TooLong", ValueError)
        raise cls(f"maximum context length is {MAX_INPUT_TOKENS + 1024} tokens, the prompt has {len(ids)}")
    with _torch.inference_mode():
        logits = _model(input_ids=_torch.tensor([ids], device=_device), logits_to_keep=1).logits[0, -1].float()
    p = logits[_label_ids[:len(options)]].softmax(-1).tolist()
    return {k: v for (k, _), v in zip(options, p)}


def choice(state, instructions, options):
    return _decide(state, instructions, list(options.items()))


def noul(state, instructions, criteria):
    crit = criteria if isinstance(criteria, dict) else {}
    yes = crit.get("true") or crit.get("yes") or "Yes."
    no = crit.get("false") or crit.get("no") or "No."
    return _decide(state, instructions, [("yes", str(yes)), ("no", str(no))])["yes"]
