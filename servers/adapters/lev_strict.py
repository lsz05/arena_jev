"""Launch the authors' own lev server (`python -m lev.serve`, github franckverrot/lev @ c48a945, servers/src/lev),
unchanged except that encoding runs with the authors' own `strict=True` flag (as lev.evaluate does): a state longer
than the 8,192-token state limit is rejected (HTTP 422 from the authors' ValueError handler) instead of being
truncated silently. Messages are prefixed with "maximum context length". Not a serve.py adapter module.

    servers/venvs/d-lev/bin/python servers/adapters/lev_strict.py --run <lev-350m snapshot dir> --port 8144
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "lev"))

import lev.model as model  # noqa: E402
import lev.serve as serve  # noqa: E402

_encode = model.DecisionModel.encode


def strict_encode(self, tok, rec, **kw):
    kw["strict"] = True
    try:
        return _encode(self, tok, rec, **kw)
    except ValueError as e:
        raise ValueError(f"maximum context length ({serve.INFER_MAX_STATE} state / {serve.INFER_MAX_BRANCH} total "
                         f"tokens): {e}") from None


model.DecisionModel.encode = strict_encode

if __name__ == "__main__":
    serve.main()
