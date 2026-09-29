"""Adapter for llm-semantic-router/decision-1.0-lex (servers/src/decision-lex = HF snapshot at ee8e74d912fc, the last revision bundling the authors'
runtime; weights identical to main). See decision_native.py for the code path."""

from decision_native import LIMIT, DecisionModel

NAME = "decision-lex"
MAX_INPUT_TOKENS = LIMIT  # pre-check on the joined request text; the exact packed check is the authors' own
_M = DecisionModel("decision-lex", "f288d873999832a3f37c6a7c4268c2ab309691e621794dbf7acab891acbbb7e6")

load = _M.load
count_tokens = _M.count_tokens
choice = _M.choice
noul = _M.noul
score = _M.score
