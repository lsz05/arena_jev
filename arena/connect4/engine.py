"""Connect Four rules engine: 7 columns x 6 rows, standard rules.

Two players drop discs in turn; a disc falls to the lowest empty cell of its column. The first player to get four of
their discs in a row (horizontally, vertically or diagonally) wins; a full board without four in a row is a draw.

Columns and rows are numbered from 1 in everything a player sees (column 1 on the left, row 1 at the bottom) and from 0
inside the engine. Player 0 moves first. An opening (a list of columns) can be played before the players take over, so
a pair of players can meet in several distinct games.
"""

from __future__ import annotations

from dataclasses import dataclass

COLS, ROWS, CONNECT = 7, 6, 4


class IllegalMove(ValueError):
    pass


@dataclass(frozen=True)
class Decision:
    player: int                 # 0 moves first
    ply: int                    # discs on the board before this move
    legal: tuple[int, ...]      # columns (0-based) that are not full


class Connect4Game:
    def __init__(self, opening: tuple[int, ...] | list[int] = ()):
        self.columns: list[list[int]] = [[] for _ in range(COLS)]  # bottom first; values are players
        self.moves: list[int] = []
        self.winner: int | None = None
        self.line: tuple[tuple[int, int], ...] = ()  # the winning cells (col, row)
        for col in opening:
            if self.over:
                raise IllegalMove(f"opening {list(opening)} ends the game")
            self.play(col)
        self.opening = tuple(opening)

    # ---- state ------------------------------------------------------------------------------------

    @property
    def ply(self) -> int:
        return len(self.moves)

    @property
    def to_move(self) -> int:
        return self.ply % 2

    @property
    def over(self) -> bool:
        return self.winner is not None or self.ply == COLS * ROWS

    def legal(self) -> tuple[int, ...]:
        return tuple(c for c in range(COLS) if len(self.columns[c]) < ROWS)

    def decision(self) -> Decision:
        if self.over:
            raise IllegalMove("the game is over")
        return Decision(self.to_move, self.ply, self.legal())

    def cell(self, col: int, row: int) -> int | None:
        column = self.columns[col]
        return column[row] if row < len(column) else None

    def landing_row(self, col: int) -> int:
        return len(self.columns[col])

    # ---- moves ------------------------------------------------------------------------------------

    def play(self, col: int) -> None:
        if self.over:
            raise IllegalMove("the game is over")
        if not 0 <= col < COLS or len(self.columns[col]) >= ROWS:
            raise IllegalMove(f"column {col + 1} is not playable")
        player = self.to_move
        self.columns[col].append(player)
        self.moves.append(col)
        line = self._line_through(col, len(self.columns[col]) - 1, player)
        if line:
            self.winner, self.line = player, line

    def wins_with(self, col: int, player: int) -> bool:
        """Would `player` complete four in a row by dropping a disc in `col` now?"""
        if len(self.columns[col]) >= ROWS:
            return False
        self.columns[col].append(player)
        try:
            return bool(self._line_through(col, len(self.columns[col]) - 1, player))
        finally:
            self.columns[col].pop()

    def _line_through(self, col: int, row: int, player: int) -> tuple[tuple[int, int], ...]:
        for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
            cells = [(col, row)]
            for sign in (1, -1):
                c, r = col + sign * dc, row + sign * dr
                while 0 <= c < COLS and 0 <= r < ROWS and self.cell(c, r) == player:
                    cells.append((c, r))
                    c, r = c + sign * dc, r + sign * dr
            if len(cells) >= CONNECT:
                return tuple(sorted(cells))
        return ()
