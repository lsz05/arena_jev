"""A TypeSafe-style POST /v1/systemone server around an in-process model adapter.

For models whose authors publish Python inference code but no /v1/systemone server. An adapter is
a module next to this file (servers/adapters/<module>.py) that uses the authors' own code and defines:

    NAME = "registry-name"
    def load(device: str) -> list:          # load the model; return the torch modules it uses (for the size count)
    def choice(state: str, instructions: str | None, options: dict[str, str]) -> dict[str, float]
        # probability for every option key (any number of options, in the given order)

and optionally:

    def noul(state, instructions, criteria: dict | None) -> float          # P(yes / true)
    def score(state, instructions, levels: list[str]) -> list[float]       # probability per level, lowest first
    def count_tokens(text: str) -> int                                     # with the model's own tokenizer
    MAX_INPUT_TOKENS = 512                                                 # the input the model can take

Without `noul` / `score`, those question types are asked through `choice` (options "true"/"false", and
"0".."L-1" for the levels). Requests whose text exceeds MAX_INPUT_TOKENS get HTTP 422 "maximum context length";
nothing is truncated silently. Model calls are serialized with a lock.

    servers/venvs/<venv>/bin/python servers/adapters/serve.py <module> --port 81xx [--device cuda]

GET /v1/models reports the name and the parameter count of the loaded modules (every tensor once), also
written to servers/logs/<name>.params.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"


class TooLong(Exception):
    pass


def count_params(modules) -> int:
    seen = {}
    for m in modules:
        for p in m.parameters():
            seen[id(p)] = p.numel()
    return sum(seen.values())


def normalize(probs: dict[str, float], keys: list[str]) -> dict[str, float]:
    missing = [k for k in keys if k not in probs]
    if missing:
        raise ValueError(f"adapter returned no probability for {missing}")
    vals = {k: float(probs[k]) for k in keys}
    if any(not math.isfinite(v) or v < 0 for v in vals.values()):
        raise ValueError(f"adapter returned invalid probabilities {vals}")
    total = sum(vals.values())
    if total <= 0:
        raise ValueError("probabilities sum to zero")
    return {k: v / total for k, v in vals.items()}


def confidence(p: dict[str, float]) -> float:
    k = len(p)
    return (max(p.values()) - 1 / k) / (1 - 1 / k) if k > 1 else 1.0


class Model:
    def __init__(self, module, device: str):
        self.m = module
        self.lock = threading.Lock()
        start = time.time()
        modules = module.load(device)
        self.params = count_params(modules or [])
        self.load_seconds = round(time.time() - start, 1)
        LOGS.mkdir(exist_ok=True)
        (LOGS / f"{module.NAME}.params.json").write_text(json.dumps(
            {"params": self.params, "method": "servers/adapters/serve.py: parameters of the modules load() returned",
             "time": time.strftime("%Y-%m-%d %H:%M:%S")}))

    def _check_length(self, text: str) -> int | None:
        count = getattr(self.m, "count_tokens", None)
        if count is None:
            return None
        with self.lock:  # adapters often count with the model's own tokenizer, which is not safe to use from two threads
            n = count(text)
        limit = getattr(self.m, "MAX_INPUT_TOKENS", None)
        if limit and n > limit:
            raise TooLong(f"maximum context length is {limit} tokens, the request has {n}")
        return n

    def answer(self, state, q: dict) -> tuple[dict, int | None]:
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        kind, instructions, criteria = q.get("type"), q.get("instructions"), q.get("criteria")
        instr = instructions if instructions is None or isinstance(instructions, str) else json.dumps(instructions)
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 1:
                raise ValueError("choice criteria must be a non-empty map")
            options = {k: (v if isinstance(v, str) or v is None else json.dumps(v)) or "" for k, v in criteria.items()}
            n = self._check_length("\n".join([state, instr or ""] + [f"{k}: {v}" for k, v in options.items()]))
            with self.lock:
                p = normalize(self.m.choice(state, instr, options), list(options))
            return {"type": "choice", "choice": max(p, key=p.get), "confidence": confidence(p), "probabilities": p}, n
        if kind == "noul":
            crit = criteria or {}
            n = self._check_length("\n".join([state, instr or ""] + [f"{k}: {v}" for k, v in crit.items()]))
            with self.lock:
                if hasattr(self.m, "noul"):
                    py = float(self.m.noul(state, instr, criteria))
                else:
                    opts = {"true": crit.get("true") or "Yes, this holds.", "false": crit.get("false") or "No, this does not hold."}
                    py = normalize(self.m.choice(state, instr, opts), ["true", "false"])["true"]
            if not (0.0 <= py <= 1.0) or not math.isfinite(py):
                raise ValueError(f"adapter returned P(yes)={py}")
            return {"type": "noul", "noul": py}, n
        if kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError("score criteria must list 2 to 10 levels")
            levels = [c if isinstance(c, str) else json.dumps(c) for c in criteria]
            n = self._check_length("\n".join([state, instr or ""] + levels))
            with self.lock:
                if hasattr(self.m, "score"):
                    vals = list(self.m.score(state, instr, levels))
                    p = normalize({str(i): v for i, v in enumerate(vals)}, [str(i) for i in range(len(levels))])
                else:
                    p = normalize(self.m.choice(state, instr, {str(i): lv for i, lv in enumerate(levels)}),
                                  [str(i) for i in range(len(levels))])
            return {"type": "score", "score": sum(int(k) * v for k, v in p.items()), "confidence": confidence(p),
                    "legend": {str(i): lv for i, lv in enumerate(levels)}, "probabilities": p}, n
        raise ValueError(f"unknown question type {kind!r}")


def make_handler(model: Model):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/v1/models"):
                return self._json(200, {"data": [{"id": model.m.NAME, "params": model.params,
                                                  "max_input_tokens": getattr(model.m, "MAX_INPUT_TOKENS", None),
                                                  "load_seconds": model.load_seconds}]})
            if self.path.startswith("/health"):
                return self._json(200, {"ok": True})
            self._json(404, {"detail": "not found"})

        def do_POST(self):  # noqa: N802
            if not self.path.startswith("/v1/systemone"):
                return self._json(404, {"detail": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                questions = body["questions"]
                if not isinstance(questions, dict) or not questions:
                    raise ValueError("questions must be a non-empty map")
                answers, tokens = {}, 0
                for name, q in questions.items():
                    answers[name], n = model.answer(body["state"], q)
                    tokens += n or 0
                self._json(200, {"model": model.m.NAME, "answers": answers,
                                 "usage": {"input_tokens": tokens, "output_tokens": 0}})
            except TooLong as e:
                self._json(422, {"detail": str(e)})
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
                self._json(422, {"detail": f"{type(e).__name__}: {e}"})
            except Exception as e:  # noqa: BLE001 - report, keep serving
                self._json(500, {"detail": f"{type(e).__name__}: {e}"})

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="adapter module name in servers/adapters/")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    model = Model(importlib.import_module(a.module), a.device)
    print(f"{model.m.NAME}: {model.params:,} parameters, loaded in {model.load_seconds}s, "
          f"serving http://{a.host}:{a.port}/v1/systemone", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(model)).serve_forever()


if __name__ == "__main__":
    main()
