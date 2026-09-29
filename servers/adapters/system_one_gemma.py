"""system-one-gemma (github akash-kamat/system-one-gemma @ cc75aa8, servers/src/system-one-gemma) with the authors' code.

Model: google/gemma-3-270m as Gemma3TextForSequenceClassification (num_labels=1), fp32, plus the LoRA scorer in
pretrained-scorer/ (PeftModel, not merged), loaded by the authors' `infer.load_trained_model`. Scoring is the authors'
`infer.score`: one sequence per option, "State:\\n<state>" + "\\n\\nQuestion:\\n<question>\\n\\nOption:\\n<option>"
(no special tokens), last-token scalar, softmax over the options with the temperature from
pretrained-scorer/metrics.json (2.35), as app.py does.

Mappings to the TypeSafe contract, fixed before any run:
  * choice: option text = the option's description when given, else the key humanized with the authors' own
    `humanize` (underscores/hyphens -> spaces), like their label options (banking77, ticket queues).
  * noul:   options "yes" / "no" exactly as in their noul training data (go_emotions, ticket language) and demos;
            true/false criteria descriptions are not used (their format has no place for them). Returns P("yes").
  * score:  the levels, in order, as the options (like their yelp stars / ticket priority score questions).
Length: the authors truncate the state (app.py max_len 256, infer.py 384). Here nothing is truncated: max_len is set
to Gemma-3-270m's 32,768-token window and longer inputs are rejected (HTTP 422). MAX_INPUT_TOKENS is a conservative
8,192 (every option is a separate full-length sequence).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent / "src" / "system-one-gemma"
NAME = "system-one-gemma"
WINDOW = 32768            # Gemma-3-270m max_position_embeddings: longest sequence the model takes
MAX_INPUT_TOKENS = 8192   # serve.py pre-check on the joined request text (conservative)

sys.path.insert(0, str(REPO))
import infer  # noqa: E402  (the authors' inference module)
from system_one import humanize  # noqa: E402

_tok = _model = None
_T = 1.0


def _too_long(msg: str) -> Exception:
    return getattr(sys.modules.get("__main__"), "TooLong", ValueError)(msg)


def load(device: str) -> list:
    global _tok, _model, _T
    _tok, _model = infer.load_trained_model(str(REPO / "pretrained-scorer"), "google/gemma-3-270m")
    _model.to(device)
    _T = float(json.loads((REPO / "pretrained-scorer" / "metrics.json").read_text()).get("temperature", 1.0))
    return [_model]


def count_tokens(text: str) -> int:
    return len(_tok(text, add_special_tokens=False)["input_ids"])


def _probs(state: str, question: str, options: list[str]) -> list[float]:
    head = len(_tok("State:\n" + state, add_special_tokens=False)["input_ids"])
    for o in options:  # infer.encode would cut the state to fit max_len; refuse instead
        tail = len(_tok("\n\nQuestion:\n" + question + "\n\nOption:\n" + o, add_special_tokens=False)["input_ids"])
        if head + tail > WINDOW:
            raise _too_long(f"maximum context length is {WINDOW} tokens per option sequence, "
                            f"this one needs {head + tail}")
    return infer.score(_model, _tok, state, question, options, max_len=WINDOW, temperature=_T)


def choice(state, instructions, options: dict) -> dict:
    texts = [v if v else humanize(k) for k, v in options.items()]
    return dict(zip(options, _probs(state, instructions or "", texts)))


def noul(state, instructions, criteria) -> float:
    return _probs(state, instructions or "", ["yes", "no"])[0]


def score(state, instructions, levels: list[str]) -> list[float]:
    return _probs(state, instructions or "", list(levels))
