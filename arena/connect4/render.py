"""What a Connect Four player is sent: a plain-text state and one choice question.

The board is drawn from the mover's point of view: X is always "you", O the opponent, so no model has to track which
color it plays. Option keys are the column numbers "1".."7" (distinct single tokens in the tokenizers used here; a key
scheme whose options share a first token makes some servers score them alike). No tactical hints: an option says where
the disc lands, not whether it wins or blocks, so the model's own reading of the board is what gets measured.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .engine import COLS, ROWS, Connect4Game, Decision

TEMPLATES = ("A", "B")  # A: the column only; B: also the row the disc lands on

INSTRUCTIONS = "Choose the column that gives you the best chance of winning this game of Connect Four."
# Optional strategy hints appended to the instructions (the default is none: the model's own reading is measured).
HINTS = {
    "defend": ("While you try to connect your own discs, you must also stop the opponent from connecting theirs: if "
               "the opponent could complete four in a row with its next disc, block that column."),
}


@dataclass
class Question:
    name: str
    instructions: str
    options: dict[str, str]  # option name -> description, in presentation order

    def to_json(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


def symbol(game: Connect4Game, col: int, row: int, you: int) -> str:
    v = game.cell(col, row)
    return "." if v is None else ("X" if v == you else "O")


def board_text(game: Connect4Game, you: int) -> str:
    lines = ["   " + " ".join(str(c + 1) for c in range(COLS))]
    for row in reversed(range(ROWS)):
        lines.append(f"{row + 1}  " + " ".join(symbol(game, c, row, you) for c in range(COLS)))
    return "\n".join(lines)


def moves_text(game: Connect4Game, you: int) -> str:
    return " ".join(("X" if i % 2 == you else "O") + str(col + 1) for i, col in enumerate(game.moves))


def render_state(game: Connect4Game, decision: Decision, with_moves: bool = True) -> str:
    you = decision.player
    lines = [
        "You are X in a game of Connect Four (7 columns, 6 rows). A disc dropped into a column falls to the lowest "
        "empty cell. The first player to get four discs in a row (across, down or diagonally) wins; a full board "
        "is a draw.",
        f"Move {game.ply + 1}: you (X) to move. Opponent: O. You {'moved first' if you == 0 else 'moved second'}.",
        "Board (row 1 at the bottom):",
        board_text(game, you),
    ]
    if with_moves and game.moves:
        lines.append(f"Moves so far, oldest first: {moves_text(game, you)}")
    return "\n".join(lines)


def question(game: Connect4Game, decision: Decision, template: str = "B", rng: random.Random | None = None,
             hint: str | None = None) -> Question:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; expected one of {TEMPLATES}")
    if hint is not None and hint not in HINTS:
        raise ValueError(f"unknown hint {hint!r}; expected one of {sorted(HINTS)}")
    options = {}
    for col in decision.legal:
        if template == "A":
            options[str(col + 1)] = f"Column {col + 1}."
        else:
            options[str(col + 1)] = f"Drop a disc in column {col + 1}: it lands on row {game.landing_row(col) + 1}."
    names = list(options)
    if rng is not None:
        rng.shuffle(names)
    instructions = INSTRUCTIONS + (" " + HINTS[hint] if hint else "")
    return Question("move", instructions, {n: options[n] for n in names})


def fit_state(game: Connect4Game, decision: Decision, q: Question, max_tokens: int | None, count_tokens) -> tuple[str, bool, int | None]:
    """The state to send, and whether the move list was kept: it is dropped (the board alone is the full position)
    only when the request would exceed the model's own input limit."""
    state = render_state(game, decision)
    if not max_tokens or count_tokens is None:
        return state, True, None
    request = lambda s: "\n".join([s, q.instructions] + [f"{k}: {v}" for k, v in q.options.items()])  # noqa: E731
    n = count_tokens(request(state))
    if n <= max_tokens or not game.moves:
        return state, True, n
    short = render_state(game, decision, with_moves=False)
    return short, False, count_tokens(request(short))
