"""Adapter for AlexWortega/openjev, subfolder qwen3.5-0.8b-nli-v2s-long (servers/adapters/serve.py).

Uses the authors' own code from the same HF repo (code/openjev_decide.py -> OpenJev.decide, on top of
code/modeling_openjev.py -> OpenJevCrossEncoder), and builds each question exactly as JevBench's reference adapter
`local_openjev` does (the path the authors used for their JevBench numbers): the labels are the option keys
(choice), "0".."L-1" (score) or ["no", "yes"] (noul), and the rubric {label: description} is appended to the
instructions as "\nAllowed answers and rubric: <json>". OpenJev.decide scores every option as one NLI hypothesis
'The answer to "<instructions>" is <label>: <description>' over the state and normalises P(entailment) over the
options. No option cap; each option costs one forward pass.

Context: the cross-encoder tokenizes "Premise: <state>\nHypothesis: <hypothesis>" with truncation at 4096 tokens
(OpenJevCrossEncoder max_len; the checkpoint is the "4k context" one), which would silently cut the end of the
pair. This adapter measures every (state, option) pair with the authors' template and tokenizer and refuses the
request (HTTP 422 "maximum context length") when one does not fit, instead of letting it truncate.
"""

from __future__ import annotations

import json
import sys

REPO = "AlexWortega/openjev"
REVISION = "26de23c44b67586b4bea31c0ef2e016e3068ae66"
SUBFOLDER = "qwen3.5-0.8b-nli-v2s-long"

NAME = "openjev-nli-0.8b"
RUBRIC_MARK = "\nAllowed answers and rubric: "     # as in jevbench local_openjev and openjev_decide

_jev = None


def load(device: str) -> list:
    global _jev
    from huggingface_hub import snapshot_download
    root = snapshot_download(REPO, revision=REVISION, allow_patterns=[f"{SUBFOLDER}/*", "code/*"])
    sys.path.insert(0, f"{root}/code")
    from openjev_decide import OpenJev
    _jev = OpenJev.from_pretrained(root, subfolder=SUBFOLDER, device=device)   # bf16, bs 32, max_len 4096
    return [_jev.ce.model]


def count_tokens(text: str) -> int:
    return len(_jev.tok(text, add_special_tokens=False)["input_ids"])


def _decide(state: str, qtype: str, instructions: str | None, labels: list[str], rubric: dict) -> list[float]:
    from openjev_decide import TEMPLATE
    q = {"type": qtype, "instructions": (instructions or "") + RUBRIC_MARK + json.dumps(rubric, ensure_ascii=False),
         "options": labels}
    # exact length check, per (state, option) pair, with the authors' own parsing, template and tokenizer
    ce = _jev.ce
    instr, crits = _jev._rubric(q["instructions"], labels)
    for w in _jev._windows(state):
        for o in labels:
            text = ce.template.format(premise=w.strip(), hypothesis=TEMPLATE.format(instr=instr, label=o, crit=crits[o]).strip())
            n = len(ce.tok(text)["input_ids"])
            if n > ce.max_len:
                raise ValueError(f"maximum context length is {ce.max_len} tokens per (state, option) pair, "
                                 f"option {o!r} needs {n}")
    (ans,) = _jev.decide(state, [q])
    if qtype == "noul":
        return [ans["noul"]]
    return [ans["probabilities"][o] for o in labels]


def choice(state, instructions, options):
    labels = list(options)
    p = _decide(state, "choice", instructions, labels, {k: (v or k) for k, v in options.items()})
    return dict(zip(labels, p))


def noul(state, instructions, criteria):
    crit = criteria or {}
    return _decide(state, "noul", instructions, ["no", "yes"],
                   {"no": crit.get("false", "No"), "yes": crit.get("true", "Yes")})[0]


def score(state, instructions, levels):
    labels = [str(i) for i in range(len(levels))]
    return _decide(state, "score", instructions, labels, dict(zip(labels, levels)))
