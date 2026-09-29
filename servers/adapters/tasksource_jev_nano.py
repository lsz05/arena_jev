"""tasksource/tasksource-jev-nano-v0 (HF @ 5ee8d27) with the model card's own PyLate snippet (pylate 1.6.0).

    model = models.ColBERT(repo)
    ctx  = model.encode([state + "\\nQuestion: " + question], is_query=False, convert_to_tensor=True)[0]
    opts = model.encode(options, is_query=True, convert_to_tensor=True)
    scores = [(o @ ctx.T).max(dim=1).values.sum() for o in opts];  probs = softmax(scores / T)

T = 0.5347228646278381 (decision_meta.json init_temperature; the card rounds it to 0.53). fp32.
Only deviation: the checkpoint's PyLate defaults truncate silently (documents to 300 tokens, options/queries to 32;
training used 512 / 64). Here document_length and query_length are both raised to ModernBERT's 8,192-token window,
so nothing is truncated (with query expansion off these lengths change nothing else), and a longer context or option
is rejected with HTTP 422.
Mappings to the TypeSafe contract, fixed before any run:
  * choice: option text = the option's description when given, else its key (the training data's options are the
    answer texts themselves).
  * noul:   the two options "yes" / "no" (the card: yes/no decisions "work the same way with two options"; the
            training data has no option texts for noul); true/false criteria descriptions are not used. P("yes").
  * score:  the levels, in order, as the options.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

NAME = "tasksource-jev-nano"
REPO = "tasksource/tasksource-jev-nano-v0"
REVISION = "5ee8d277a830de19c7cf2329be425a2c9d3475eb"
WINDOW = 8192             # ModernBERT max_position_embeddings (prefix token + [CLS] ... [SEP])
MAX_INPUT_TOKENS = 8000   # serve.py pre-check on the joined request text

_model = None
_T = 1.0


def _too_long(msg: str) -> Exception:
    return getattr(sys.modules.get("__main__"), "TooLong", ValueError)(msg)


def load(device: str) -> list:
    global _model, _T
    from huggingface_hub import snapshot_download
    from pylate import models
    path = snapshot_download(REPO, revision=REVISION)
    _model = models.ColBERT(path, device=device, document_length=WINDOW, query_length=WINDOW)
    _model.eval()
    _T = float(json.loads((Path(path) / "decision_meta.json").read_text())["init_temperature"])
    return [_model]


def count_tokens(text: str) -> int:
    return len(_model.tokenizer(text, add_special_tokens=False)["input_ids"])


def _check(text: str, what: str) -> None:
    n = len(_model.tokenizer(text)["input_ids"]) + 1  # + the [D]/[Q] prefix token PyLate inserts
    if n > WINDOW:
        raise _too_long(f"maximum context length is {WINDOW} tokens, the {what} has {n}")


@torch.no_grad()
def _probs(state: str, question: str, options: list[str]) -> list[float]:
    context = state + "\nQuestion: " + question
    _check(context, "state + question")
    for o in options:
        _check(o, "option")
    ctx = _model.encode([context], is_query=False, convert_to_tensor=True, show_progress_bar=False)[0]
    opts = _model.encode(options, is_query=True, convert_to_tensor=True, show_progress_bar=False)
    scores = torch.stack([(o @ ctx.T).max(dim=1).values.sum() for o in opts])
    return torch.softmax(scores.float() / _T, dim=0).cpu().tolist()


def choice(state, instructions, options: dict) -> dict:
    texts = [v if v else k for k, v in options.items()]
    return dict(zip(options, _probs(state, instructions or "", texts)))


def noul(state, instructions, criteria) -> float:
    return _probs(state, instructions or "", ["yes", "no"])[0]


def score(state, instructions, levels: list[str]) -> list[float]:
    return _probs(state, instructions or "", list(levels))
