"""Adapter for fastino/GLiNER2.5-Decide (GLiNER2 span architecture, DeBERTa-v3-large encoder).

Code path: the authors' `gliner2` package 2.0.0 (pip, [local] extra) in venvs/c-gliner, checkpoint
servers/src/gliner2.5-decide = HF snapshot 5a7adf72a23b. AutoExtractor.from_pretrained(dir, map_location=device),
FP32, no quantize/compile (package defaults).

Mapping (as the JevBench maintainers' gliner2_local adapter, except the question slot): one classification task
"decision" per question; labels carry descriptions ({label: description}) as in the card's "Labels with a
description" example; the instructions go into the task's `prompt` key, the card's documented slot for a
question over a passage ("Question over a passage": {"labels": [...], "prompt": "..."}); the text is the state.
  choice: {option: description (or the option name when empty)}
  noul:   {"yes": criteria.true or "Yes", "no": criteria.false or "No"}  -> P(yes)
          (without criteria: plain labels ["yes", "no"], the card's yes/no gate)
  score:  {"0": level 0 text, "1": ...}
Distribution: the single-label head's softmax over all labels, read out with the package's documented
class_act="softmax", multi_label=True, cls_threshold=0.0, include_confidence=True (same numbers as the default
single-label decode for the argmax).

Length: the package imposes no input limit and never truncates (max_len=None); the DeBERTa-v3 encoder uses
relative positions. MAX_INPUT_TOKENS below is only a server memory guard.
"""

from __future__ import annotations

from pathlib import Path

NAME = "gliner2.5-decide"
CKPT = Path(__file__).resolve().parent.parent / "src" / "gliner2.5-decide"
MAX_INPUT_TOKENS = 8192

_ex = None


def load(device: str) -> list:
    global _ex
    from gliner2 import AutoExtractor

    _ex = AutoExtractor.from_pretrained(str(CKPT), map_location=device)
    _ex.eval()
    return [_ex]


def count_tokens(text: str) -> int:
    tok = _ex.processor.tokenizer if hasattr(_ex, "processor") else _ex.tokenizer
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _classify(state: str, instructions, labels: dict) -> dict:
    kwargs = dict(multi_label=True, cls_threshold=0.0, class_act="softmax")
    if instructions:
        kwargs["prompt"] = instructions
    schema = _ex.create_schema().classification("decision", labels, **kwargs)
    out = _ex.extract(state, schema, include_confidence=True)
    probs = {str(d["label"]): float(d["confidence"]) for d in out["decision"]}
    if set(probs) != set(labels if isinstance(labels, list) else labels.keys()):
        raise RuntimeError(f"labels returned {sorted(probs)} != asked {sorted(labels)}")
    return probs


def choice(state, instructions, options: dict) -> dict:
    if len(options) == 1:
        return {k: 1.0 for k in options}
    return _classify(state, instructions, {k: (v or k) for k, v in options.items()})


def noul(state, instructions, criteria) -> float:
    crit = criteria or {}
    if crit.get("true") or crit.get("false"):
        labels = {"yes": crit.get("true") or "Yes", "no": crit.get("false") or "No"}
    else:  # the card's yes/no gate: plain labels, no descriptions
        labels = ["yes", "no"]
    p = _classify(state, instructions, labels)
    return p["yes"] / (p["yes"] + p["no"])


def score(state, instructions, levels: list) -> list:
    p = _classify(state, instructions, {str(i): lv for i, lv in enumerate(levels)})
    return [p[str(i)] for i in range(len(levels))]
