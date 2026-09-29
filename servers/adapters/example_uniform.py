"""Example adapter for servers/adapters/serve.py: uniform probabilities (no model). Used to test the server."""

NAME = "example-uniform"
MAX_INPUT_TOKENS = 100_000


def load(device: str) -> list:
    return []


def count_tokens(text: str) -> int:
    return len(text) // 4


def choice(state, instructions, options):
    return {k: 1.0 for k in options}
