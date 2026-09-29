"""Adapter for llm-semantic-router/Decision-1.0-Kai (servers/src/decision-kai = HF snapshot at 7185f514f54b, the last revision bundling the authors'
runtime; weights identical to main). See decision_native.py for the code path."""

from decision_native import LIMIT, DecisionModel

NAME = "decision-kai"
MAX_INPUT_TOKENS = LIMIT  # pre-check on the joined request text; the exact packed check is the authors' own
_M = DecisionModel("decision-kai", "c1bf07ab1c4c3fa1f819256d3de858d1ed87869bdfa663553280d7e78b88bee4")

load = _M.load
count_tokens = _M.count_tokens
choice = _M.choice
noul = _M.noul
score = _M.score
