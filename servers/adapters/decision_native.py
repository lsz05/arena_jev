"""Shared adapter code for the Decision 1.0 Kai / Lex encoders (llm-semantic-router), used by decision_kai.py and
decision_lex.py. Not an adapter itself (no NAME).

Code path: the authors' own local runtime, bundled in the HF repo at the last revision that shipped it
(Kai 7185f514f54b, Lex ee8e74d912fc; the weight/config/tokenizer files are byte-identical to the current `main`,
which is a model-only release). We call `decision_inference.SystemOne` (their System One request -> answer API:
default B8 typed scheduling, FP32, complete 1,024-token admission, no truncation) on `decision_runtime.load_native`.

Only deviation: the authors' loader refuses non-ROCm devices (`_amd_device`: "this release does not supply a
CPU/NVIDIA inference path"). We accept a CUDA device instead; on CUDA their ModernBERT SDPA layout guard falls back
to the unmodified upstream attention (their own code path for non-HIP). TF32 is disabled as in their systemone.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
LIMIT = 1024  # complete packed tokens (state + question + candidates + markers/special tokens), authors' profile


def _too_long(msg: str) -> Exception:
    main = sys.modules.get("__main__")
    cls = getattr(main, "TooLong", ValueError)
    return cls(msg)


def _maybe_structured(text):
    """serve.py hands structured states over as json.dumps(obj, ensure_ascii=False). Give the authors' code the
    object back so it applies its own canonical form (compact JSON, sorted keys)."""
    if isinstance(text, str) and text[:1] in "{[":
        try:
            obj = json.loads(text)
        except ValueError:
            return text
        if isinstance(obj, (dict, list)) and json.dumps(obj, ensure_ascii=False) == text:
            return obj
    return text


class DecisionModel:
    def __init__(self, repo_dir: str, manifest_sha256: str):
        self.root = SRC / repo_dir
        self.manifest = manifest_sha256
        self.client = None
        self.tokenizer = None

    def load(self, device: str) -> list:
        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch
        import decision_runtime.native as rt
        from decision_inference import SystemOne

        def _cuda_device(dev):  # replaces the ROCm-only guard; everything else unchanged
            selected = torch.device(dev)
            if selected.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("a CUDA device is required")
            if selected.index is None:
                selected = torch.device("cuda", torch.cuda.current_device())
            return selected

        rt._amd_device = _cuda_device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.mha.set_fastpath_enabled(False)
        dev = "cuda:0" if device == "cuda" else device
        native = rt.load_native(str(self.root / "native"), expected_manifest_sha256=self.manifest, device=dev)
        self.client = SystemOne(native)  # default batching (typed B8), public model name from the manifest
        self.tokenizer = native.collator.tokenizer
        return [native.model]

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    def _ask(self, state, question: dict) -> dict:
        try:
            out = self.client.system_one(state=_maybe_structured(state), questions={"decision": question})
        except ValueError as e:
            msg = str(e)
            if "exceeds" in msg and "tokens" in msg or "no room for state" in msg:
                raise _too_long(f"maximum context length is {LIMIT} tokens (complete packed input): {msg}") from e
            raise
        return out["answers"]["decision"]

    def choice(self, state, instructions, options: dict) -> dict:
        if len(options) == 1:  # the authors' API needs >= 2 candidates; a single option is the only answer
            return {k: 1.0 for k in options}
        crit = {k: (v if v else None) for k, v in options.items()}  # serve.py maps null to ""; authors use null
        ans = self._ask(state, {"type": "choice", "instructions": _maybe_structured(instructions), "criteria": crit})
        return {k: float(ans["probabilities"][k]) for k in options}

    def noul(self, state, instructions, criteria) -> float:
        q = {"type": "noul", "instructions": _maybe_structured(instructions)}
        if criteria:
            q["criteria"] = {k: v for k, v in criteria.items() if k in ("true", "false") and v}
            if not q["criteria"]:
                del q["criteria"]
        return float(self._ask(state, q)["noul"])

    def score(self, state, instructions, levels: list) -> list:
        ans = self._ask(state, {"type": "score", "instructions": _maybe_structured(instructions),
                                "criteria": [_maybe_structured(lv) for lv in levels]})
        return [float(ans["probabilities"][str(i)]) for i in range(len(levels))]
