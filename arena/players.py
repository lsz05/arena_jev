"""Players: each turns (observation, decision) into one legal action.

- RandomPlayer: uniform over the moves (each playable card, or draw/keep), random wild color.
- HeuristicPlayer: a fixed, simple strategy used as a reference opponent.
- SystemOnePlayer: asks a TypeSafe-style decision model; works with Jev and every local server
  that speaks POST /v1/systemone.

Every player gets a per-decision `rng` from the match runner, so games are reproducible.
"""

from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass, field

from .systemone import SystemOneClient, SystemOneError
from .uno.cards import COLORS, DRAW_TWO, REVERSE, SKIP, WILD_DRAW_FOUR, Card, Color
from .uno.engine import CHALLENGE, START_COLOR, TURN, Accept, Action, Challenge, ChooseColor, Decision, Draw, Keep, Observation, Play
from .uno import render


@dataclass
class Choice:
    action: Action
    info: dict = field(default_factory=dict)  # logged with the decision: model calls, fallbacks, ...


class Player:
    name: str

    def act(self, obs: Observation, decision: Decision, rng: random.Random) -> Choice:
        raise NotImplementedError


def _pass(decision: Decision) -> Action:
    return Draw() if decision.kind == TURN else Keep()


def _most_held_color(hand: tuple[Card, ...] | list[Card], rng: random.Random) -> Color:
    counts = Counter(c.color for c in hand if c.color)
    if not counts:
        return rng.choice(COLORS)
    best = max(counts.values())
    return next(c for c in COLORS if counts[c] == best)


class RandomPlayer(Player):
    def __init__(self, name: str = "random"):
        self.name = name

    def act(self, obs: Observation, decision: Decision, rng: random.Random) -> Choice:
        if decision.kind == START_COLOR:
            return Choice(ChooseColor(rng.choice(COLORS)))
        if decision.kind == CHALLENGE:
            return Choice(rng.choice([Challenge(), Accept()]))
        card = rng.choice(list(decision.playable) + ([] if decision.must_play and decision.playable else [None]))
        if card is None:
            return Choice(_pass(decision))
        return Choice(Play(card, rng.choice(COLORS) if card.is_wild else None))


class HeuristicPlayer(Player):
    """Never draws when it can play; saves wilds for when nothing else fits; attacks with
    Draw Two / Skip / Reverse / Wild Draw Four when the next player holds two cards or fewer;
    otherwise plays the color it holds most of, highest points first; names its most-held color.
    Never bluffs a Wild Draw Four; challenges one when the player who played it still holds 6+ cards."""

    CHALLENGE_MIN_CARDS = 6

    ATTACK_ORDER = (DRAW_TWO, SKIP, REVERSE)

    def __init__(self, name: str = "heuristic"):
        self.name = name

    def act(self, obs: Observation, decision: Decision, rng: random.Random) -> Choice:
        if decision.kind == START_COLOR:
            return Choice(ChooseColor(_most_held_color(obs.hand, rng)))
        if decision.kind == CHALLENGE:
            offender = render.last_wild_draw_four(obs)[0]
            return Choice(Challenge() if obs.hand_sizes[offender] >= self.CHALLENGE_MIN_CARDS else Accept())
        honest = tuple(c for c in decision.playable
                       if not (c.rank == WILD_DRAW_FOUR and any(h.color == obs.color for h in obs.hand)))
        if not honest:
            return Choice(_pass(decision))
        card = self._pick(obs, Decision(decision.player, decision.kind, honest, decision.must_play))
        rest = list(obs.hand)
        rest.remove(card)
        return Choice(Play(card, _most_held_color(rest, rng) if card.is_wild else None))

    def _pick(self, obs: Observation, decision: Decision) -> Card:
        playable = decision.playable
        colored = [c for c in playable if not c.is_wild]
        threat = obs.hand_sizes[obs.seat_after(1)] <= 2
        if threat:
            for rank in self.ATTACK_ORDER:
                hit = [c for c in colored if c.rank == rank]
                if hit:
                    return hit[0]
            wd4 = [c for c in playable if c.rank == WILD_DRAW_FOUR]
            if wd4:
                return wd4[0]
        if colored:
            counts = Counter(c.color for c in obs.hand if c.color)
            return max(colored, key=lambda c: (counts[c.color], c.points))
        plain = [c for c in playable if c.rank != WILD_DRAW_FOUR]
        return plain[0] if plain else playable[0]


class SystemOnePlayer(Player):
    """Asks a decision model through POST /v1/systemone, one question per request.

    The move is asked first; the color only if a wild was chosen. A question with a single option
    (no playable card: draw) is not sent. `policy` "argmax" takes the most
    probable option (ties: first as presented), "sample" draws from the returned distribution.
    Options above `max_options` are decided by knockout: groups first, then the group winners.
    On an error the player falls back to a random legal move and the decision is flagged.
    """

    def __init__(self, name: str, client: SystemOneClient, *, template: str = "B", history_rounds: int = 1,
                 policy: str = "argmax", max_options: int = 255, max_prompt_tokens: int | None = None,
                 count_tokens=None, max_option_tokens: int | None = None):
        if policy not in ("argmax", "sample"):
            raise ValueError(f"policy must be argmax or sample, got {policy!r}")
        self.name = name
        self.client = client
        self.template = template
        self.history_rounds = history_rounds
        self.policy = policy
        self.max_options = max_options
        self.max_prompt_tokens = max_prompt_tokens  # history is trimmed (oldest round first) to fit this budget
        self.count_tokens = count_tokens
        self.max_option_tokens = max_option_tokens  # an option above this (model's tokenizer) is shortened

    def act(self, obs: Observation, decision: Decision, rng: random.Random) -> Choice:
        info: dict = {"calls": []}
        try:
            if decision.kind == START_COLOR:
                q = render.color_question(obs, decision, None, self.template, rng)
                return Choice(ChooseColor(Color(self._ask(obs, decision, q, rng, info))), info)
            if decision.kind == CHALLENGE:
                q = render.challenge_question(obs, decision, self.template, rng)
                return Choice(render.challenge_action(self._ask(obs, decision, q, rng, info)), info)
            q = render.move_question(obs, decision, self.template, rng)
            card = render.move_to_card(decision, self._ask(obs, decision, q, rng, info))
            if card is None:
                return Choice(_pass(decision), info)
            color = None
            if card.is_wild:
                q = render.color_question(obs, decision, card, self.template, rng)
                color = Color(self._ask(obs, decision, q, rng, info))
            return Choice(Play(card, color), info)
        except (SystemOneError, KeyError, ValueError) as e:
            info["fallback"] = True
            info["error"] = f"{type(e).__name__}: {e}"
            fallback = RandomPlayer().act(obs, decision, rng)
            return Choice(fallback.action, info)

    def _ask(self, obs: Observation, decision: Decision, q: render.Question, rng: random.Random, info: dict) -> str:
        names = list(q.options)
        if len(names) == 1:  # nothing to decide (e.g. no playable card: draw); some servers reject 1-option questions
            info["forced"] = True
            return names[0]
        if len(names) > self.max_options:
            groups = [names[i:i + self.max_options] for i in range(0, len(names), self.max_options)]
            winners = [g[0] if len(g) == 1 else self._ask(obs, decision, _subset(q, g), rng, info) for g in groups]
            return self._ask(obs, decision, _subset(q, winners), rng, info)
        compacted = {}
        if self.max_option_tokens and self.count_tokens:
            q, compacted = render.compact_question(q, lambda t: self.count_tokens(t) <= self.max_option_tokens)
        state, rounds, n_tokens = render.fit_state(obs, decision, q, self.history_rounds, self.max_prompt_tokens,
                                                   self.count_tokens)
        start = time.perf_counter()
        response = self.client.evaluate(state, {q.name: q.to_json()})
        latency_ms = (time.perf_counter() - start) * 1000
        probs = response["answers"][q.name]["probabilities"]
        missing = set(names) - set(probs)
        if missing:
            raise SystemOneError(f"response lacks probabilities for {sorted(missing)}")
        p = [max(float(probs[n]), 0.0) for n in names]
        pick = names[_sample(p, rng) if self.policy == "sample" else p.index(max(p))]
        info["calls"].append({
            "question": q.name,
            "options": names,
            "probabilities": {n: float(probs[n]) for n in names},
            "pick": pick,
            "history_rounds": rounds,
            "prompt_tokens": n_tokens,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": (response.get("usage") or {}).get("input_tokens"),
            "server_model": response.get("model"),
            **({"compacted": compacted} if compacted else {}),
        })
        return pick


def _subset(q: render.Question, names: list[str]) -> render.Question:
    return render.Question(q.name, q.instructions, {n: q.options[n] for n in names})


def _sample(weights: list[float], rng: random.Random) -> int:
    total = sum(weights)
    if total <= 0:
        return rng.randrange(len(weights))
    x = rng.random() * total
    for i, w in enumerate(weights):
        x -= w
        if x < 0:
            return i
    return len(weights) - 1
