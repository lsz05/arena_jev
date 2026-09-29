"""Launch the authors' own TypeSafe server for Lumma-Fev models (`lumma-fev-serve`, pip lumma-fev==0.1.1), unchanged
except for one guard: a state longer than the model's state limit is rejected with ValueError (the authors' server
answers HTTP 422) instead of being truncated silently by FevForDecision.encode. Not a serve.py adapter module.

    servers/venvs/d-lumma/bin/python servers/adapters/lumma_fev_guarded.py --model FrontiersMind/Lumma-fev-0.1b \
        --revision <commit> --port 8141            (all arguments are lumma-fev-serve's own)
"""

import sys

import lumma_fev.server as server
from lumma_fev.local import load as authors_load


def guarded_load(*args, **kwargs):
    model = authors_load(*args, **kwargs)
    render = sys.modules[type(model).__module__].render   # the authors' own state rendering
    encode = model.encode

    def checked_encode(tok, state, packed):
        max_state, max_row = model.limits()
        n = 1 + len(model.text_ids(tok, render(state)))  # <state> delimiter + state tokens, exactly as encode builds it
        if n > max_state:
            raise ValueError(f"maximum context length: the state has {n} tokens, the model reads at most {max_state}")
        try:
            return encode(tok, state, packed)
        except ValueError as e:                            # a question row over the row limit (authors' own check)
            raise ValueError(f"maximum context length ({max_row} tokens per row): {e}") from None

    model.encode = checked_encode
    return model


server.load = guarded_load

if __name__ == "__main__":
    server.main()
