"""Truncation guard for models that build their input with `laya.common.build_sequence` (Laya, Lavoir).

Laya's sequence builder never refuses: it caps each option at 48 tokens, compresses the options when question +
options exceed `head_max_len`, cuts the instruction, and cuts the state to fit `max_len`, all silently. The arena
wants over-long requests refused, so before every call we build the sequence twice with laya's own function:
once with the model's real budgets and once with unlimited budgets. If they differ (or an option is over the
48-token cap), something would have been cut, and the request is refused as too long. Nothing about the
sequence the model sees is changed.
"""

from __future__ import annotations

import sys

OPTION_CAP = 48  # laya.common.build_sequence: each option is cut to 48 tokens


def too_long(msg: str) -> Exception:
    cls = getattr(sys.modules.get("__main__"), "TooLong", ValueError)
    return cls(msg)


def check(tok, state, q: dict, max_len: int, head_max_len: int) -> int:
    """Raise TooLong if laya would truncate anything; return the packed length otherwise."""
    from laya.common import build_sequence, encode_text, render_options

    mask = tok.mask_token
    for opt in render_options(q):
        n = len(encode_text(tok, " " + opt.replace(mask, " "), add_special_tokens=False)["input_ids"])
        if n > OPTION_CAP:
            raise too_long(f"maximum context length: an option has {n} tokens, the model's cap is {OPTION_CAP}")
    left = isinstance(state, list)
    full, _ = build_sequence(tok, state, q, 10**9, 10**9, truncate_left=left)
    real, _ = build_sequence(tok, state, q, max_len, head_max_len, truncate_left=left)
    if real != full:
        head = full.index(tok.sep_token_id, full.index(tok.sep_token_id) + 1) + 1  # through the options' [SEP]
        raise too_long(f"maximum context length is {max_len} tokens (question + options at most {head_max_len}), "
                       f"the request needs {len(full)} (question + options {head - 3})")
    return len(full)
