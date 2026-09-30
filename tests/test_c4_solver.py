"""The C solver against an independent brute-force search (full game tree, no pruning) on late positions."""
import random
from functools import lru_cache

from arena.connect4 import solver
from arena.connect4.engine import COLS, ROWS, Connect4Game


def sign(v):
    return (v > 0) - (v < 0)


def brute(moves: tuple) -> int:
    """+1 / 0 / -1: the outcome for the side to move with perfect play from both sides."""
    @lru_cache(maxsize=None)
    def value(ms: tuple) -> int:
        g = Connect4Game(list(ms))
        best = -1
        for c in g.legal():
            if g.wins_with(c, g.to_move):
                return 1
            nxt = ms + (c,)
            v = 0 if len(nxt) == COLS * ROWS else -value(nxt)
            best = max(best, v)
            if best == 1:
                break
        return best
    return value(moves)


def late_positions(n, ply, seed):
    rng, out = random.Random(seed), []
    while len(out) < n:
        g = Connect4Game()
        while not g.over and g.ply < ply:
            quiet = [c for c in g.legal() if not g.wins_with(c, g.to_move)] or g.legal()
            g.play(rng.choice(quiet))
        if not g.over and g.ply == ply:
            out.append(tuple(g.moves))
    return out


def test_every_move_matches_brute_force():
    for ply in (34, 32, 30):
        for moves in late_positions(12, ply, seed=ply):
            g = Connect4Game(list(moves))
            got = solver.analyze(list(moves))
            for c in range(COLS):
                if c not in g.legal():
                    assert got[c] is None
                    continue
                if g.wins_with(c, g.to_move):
                    expected = 1
                else:
                    nxt = moves + (c,)
                    expected = 0 if len(nxt) == COLS * ROWS else -brute(nxt)
                assert sign(got[c]) == expected, (moves, c, got, expected)
            assert sign(solver.solve(list(moves))) == brute(moves)


def test_invalid_positions_are_refused():
    import pytest
    with pytest.raises(ValueError):
        solver.analyze([0, 1, 0, 1, 0, 1, 0, 1])  # the first player has already won
