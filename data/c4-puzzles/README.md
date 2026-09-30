# Connect Four decision puzzles

1000 Connect Four positions, each with exactly one correct column, made by `python -m arena c4-puzzles-make`
(`arena/connect4/puzzles.py`, seed 20261001) and asked as single decisions by `python -m arena c4-puzzles-run`.

| File | Type | Tier | Puzzles | The correct column |
|---|---|---|---|---|
| `win.jsonl` | win | easy | 250 | wins at once; the opponent has no immediate win |
| `block.jsonl` | block | medium | 250 | blocks the opponent's only immediate win (the mover cannot win at once; the block does not lose on top) |
| `fork.jsonl` | fork | hard | 250 | wins in two moves: the opponent cannot stop every threat, and the solver confirms it is the only winning move |
| `solver.jsonl` | solver | hard | 250 | none of the above; by a perfect solver it is the only move that keeps the result (155 keep a win, 95 a draw) |

- Positions come from random games in which players avoid completing four when they can, so threats pile up.
- No two puzzles are copies or left-right mirrors of each other.
- In every type the correct column is spread evenly over the seven columns (35 or 36 each), and win / block are split
  evenly between horizontal, vertical and diagonal fours.
- Every answer is verified by the Connect Four engine, and fork / solver answers also by the C solver
  (`arena/connect4/solver.c`, checked against a brute-force search).

One puzzle per line: `id`, `family` (the type), `tier`, `moves` (the columns played so far, 1-based), `ply`,
`mover` (first or second player), `state` and `question` (exactly what a model is sent: the board from the mover's
side, where X is "you", and a choice over the legal columns, each saying where the disc lands), `labels`, `expected`
(the correct column), `detail` (the four made or stopped, the threats after a fork, or the solver's result for every
column) and `provenance`.
