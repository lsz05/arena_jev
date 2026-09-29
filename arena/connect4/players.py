"""Connect Four players: random, a simple rule-based heuristic, and System One models."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ..systemone import SystemOneClient, SystemOneError
from . import render
from .engine import Connect4Game, Decision


@dataclass
class Choice:
    col: int                                    # 0-based column
    info: dict = field(default_factory=dict)    # logged with the decision: model calls, fallbacks, ...


class RandomPlayer:
    def __init__(self, name: str = "random"):
        self.name = name

    def act(self, game: Connect4Game, decision: Decision, rng: random.Random) -> Choice:
        return Choice(rng.choice(decision.legal))


class HeuristicPlayer:
    """Wins if it can, blocks an immediate win of the opponent, avoids moves that let the opponent win on top,
    otherwise prefers central columns."""

    def __init__(self, name: str = "heuristic"):
        self.name = name

    def act(self, game: Connect4Game, decision: Decision, rng: random.Random) -> Choice:
        me, opp = decision.player, 1 - decision.player
        for c in decision.legal:
            if game.wins_with(c, me):
                return Choice(c)
        for c in decision.legal:
            if game.wins_with(c, opp):
                return Choice(c)
        safe = [c for c in decision.legal if not gives_win_on_top(game, c, me)]
        pool = safe or list(decision.legal)
        return Choice(min(pool, key=lambda c: (abs(c - 3), rng.random())))


def gives_win_on_top(game: Connect4Game, col: int, me: int) -> bool:
    """After `me` plays `col`, could the opponent win by playing on top of it?"""
    game.columns[col].append(me)
    try:
        return game.wins_with(col, 1 - me)
    finally:
        game.columns[col].pop()


class SystemOnePlayer:
    """A model behind POST /v1/systemone. One choice question per move; the options are shuffled with a seed; a move
    with a single legal column is taken without asking. `policy` "argmax" takes the most probable column (ties: first
    as presented), "sample" draws from the returned distribution. More columns than `max_options` are decided by
    knockout (groups first, then the group winners). On an error the player falls back to a random legal column and
    the decision is flagged."""

    def __init__(self, name: str, client: SystemOneClient, *, template: str = "B", policy: str = "argmax",
                 max_options: int = 255, max_prompt_tokens: int | None = None, count_tokens=None,
                 hint: str | None = None):
        if policy not in ("argmax", "sample"):
            raise ValueError(f"policy must be argmax or sample, got {policy!r}")
        self.name, self.client, self.template, self.policy = name, client, template, policy
        self.max_options = max_options
        self.max_prompt_tokens = max_prompt_tokens
        self.count_tokens = count_tokens
        self.hint = hint

    def act(self, game: Connect4Game, decision: Decision, rng: random.Random) -> Choice:
        info: dict = {"calls": []}
        if len(decision.legal) == 1:
            info["forced"] = True
            return Choice(decision.legal[0], info)
        q = render.question(game, decision, self.template, rng, self.hint)
        try:
            return Choice(int(self._ask(game, decision, q, rng, info)) - 1, info)
        except (SystemOneError, KeyError, ValueError) as e:
            info["fallback"] = True
            info["error"] = f"{type(e).__name__}: {e}"
            return Choice(rng.choice(decision.legal), info)

    def _ask(self, game: Connect4Game, decision: Decision, q: render.Question, rng: random.Random, info: dict) -> str:
        names = list(q.options)
        if len(names) == 1:
            return names[0]
        if len(names) > self.max_options:
            groups = [names[i:i + self.max_options] for i in range(0, len(names), self.max_options)]
            winners = [g[0] if len(g) == 1 else self._ask(game, decision, _subset(q, g), rng, info) for g in groups]
            return self._ask(game, decision, _subset(q, winners), rng, info)
        state, with_moves, n_tokens = render.fit_state(game, decision, q, self.max_prompt_tokens, self.count_tokens)
        start = time.perf_counter()
        response = self.client.evaluate(state, {q.name: q.to_json()})
        latency_ms = (time.perf_counter() - start) * 1000
        probs = response["answers"][q.name]["probabilities"]
        missing = set(names) - set(probs)
        if missing:
            raise SystemOneError(f"response lacks probabilities for {sorted(missing)}")
        p = [max(float(probs[n]), 0.0) for n in names]
        if self.policy == "sample" and sum(p) > 0:
            pick = rng.choices(names, weights=p)[0]
        else:
            pick = names[p.index(max(p))]
        info["calls"].append({
            "question": q.name,
            "options": names,
            "probabilities": {n: float(probs[n]) for n in names},
            "pick": pick,
            "with_moves": with_moves,
            "prompt_tokens": n_tokens,
            "latency_ms": round(latency_ms, 1),
            "server_model": response.get("model"),
        })
        return pick


def _subset(q: render.Question, names: list[str]) -> render.Question:
    return render.Question(q.name, q.instructions, {n: q.options[n] for n in names})
