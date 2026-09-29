"""Token counting for prompt budgets: each model's own tokenizer (a Hugging Face repo id, or a local tokenizer.json or
directory holding one), against that model's own input limit."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"


@lru_cache(maxsize=None)
def counter(name: str = DEFAULT_TOKENIZER):
    """A thread-safe callable text -> number of tokens (no special tokens), for a Hugging Face tokenizer."""
    import tokenizers

    os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    path = Path(os.path.expandvars(name)).expanduser()
    if path.is_dir():
        path = path / "tokenizer.json"
    tok = tokenizers.Tokenizer.from_file(str(path)) if path.is_file() else tokenizers.Tokenizer.from_pretrained(name)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)
