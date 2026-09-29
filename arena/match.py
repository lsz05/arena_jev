"""Play one game and produce its records.

Each decision yields one record (what the player saw, what it chose, model calls, diagnostics).
The game record keeps the full, unhidden event log and starting hands, enough to replay the game.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace

from .players import Player
from .uno.cards import WILD_DRAW_FOUR
from .uno.engine import (CHALLENGE, DRAWN_CARD, TURN, Challenge, Decision, Draw, Event, IllegalAction, Keep,
                          Observation, Play, UnoGame, action_to_json)


def presented(game: UnoGame, player: int, seed: int, shuffle_hand: bool) -> Observation:
    """The observation a player gets; with `shuffle_hand` its hand is listed in a seeded random order
    (its own RNG stream, so option shuffles and replays are unaffected)."""
    obs = game.observation(player)
    if not shuffle_hand:
        return obs
    hand = list(obs.hand)
    random.Random(f"{seed}:{game.steps}:hand").shuffle(hand)
    return replace(obs, hand=tuple(hand))


def play_game(players: Sequence[Player], seed: int, *, game_id: str, on_decision: Callable[[dict], None] | None = None,
              max_steps: int = 5000, meta: dict | None = None, wd4_challenge: bool = False,
              shuffle_hand: bool = False, must_play: bool = False) -> dict:
    game = UnoGame(len(players), seed, max_steps=max_steps, wd4_challenge=wd4_challenge, must_play=must_play)
    while not game.over:
        decision = game.decision()
        player = players[decision.player]
        obs = presented(game, decision.player, seed, shuffle_hand)
        guilty = game.pending_challenge["guilty"] if decision.kind == CHALLENGE else None
        rng = random.Random(f"{seed}:{game.steps}")
        choice = player.act(obs, decision, rng)
        if choice.action not in decision.legal_actions():
            raise IllegalAction(f"{player.name} chose {choice.action} for {decision}")
        if on_decision is not None:
            on_decision({
                "game": game_id,
                "step": game.steps,
                "turn": game.turn,
                "seat": decision.player,
                "player": player.name,
                "kind": decision.kind,
                "hand": [c.id for c in obs.hand],
                "top": obs.top_card.id,
                "color": obs.color.value if obs.color else None,
                "hand_sizes": list(obs.hand_sizes),
                "playable": [c.id for c in decision.playable],
                "action": action_to_json(choice.action),
                "diag": diagnostics(obs, decision, choice.action, guilty),
                **choice.info,
            })
        game.step(choice.action)

    result = game.result()
    names = [p.name for p in players]
    return {
        "game": game_id,
        "seed": seed,
        **(meta or {}),
        "seats": names,
        "winner_seat": result.winner,
        "winner": names[result.winner] if result.winner is not None else None,
        "score": result.score,
        "hand_points": list(result.hand_points),
        "turns": result.turns,
        "steps": result.steps,
        "aborted": result.winner is None,
        "rules": {"wd4_challenge": wd4_challenge, "shuffle_hand": shuffle_hand, "max_steps": max_steps,
                  "must_play": must_play},
        "initial_hands": [[c.id for c in h] for h in game.initial_hands],
        "events": [event_to_json(e) for e in game.log],
    }


def diagnostics(obs: Observation, decision: Decision, action, guilty: bool | None = None) -> dict:
    """Per-decision flags for obviously questionable moves (not errors: all moves are legal).
    For a challenge decision `guilty` is the hidden truth, logged to score the decision afterwards."""
    if decision.kind == CHALLENGE:
        challenged = isinstance(action, Challenge)
        return {"challenged": challenged, "guilty": guilty, "challenge_right": challenged == guilty}
    wd4 = isinstance(action, Play) and action.card.rank == WILD_DRAW_FOUR
    if decision.kind == DRAWN_CARD:
        out = {"kept_playable": isinstance(action, Keep)}
        if wd4 and obs.wd4_challenge:
            out["wd4_bluff"] = any(c.color == obs.color for c in obs.hand)
        return out
    if decision.kind != TURN:
        return {}
    colored_playable = any(not c.is_wild for c in decision.playable)
    diag = {
        "could_play": bool(decision.playable),
        "drew_with_playable": isinstance(action, Draw) and bool(decision.playable),
    }
    if wd4 and obs.wd4_challenge:
        diag["wd4_bluff"] = any(c.color == obs.color for c in obs.hand)
    if isinstance(action, Play) and action.card.is_wild:
        diag["wild_with_colored"] = colored_playable
        rest = list(obs.hand)
        rest.remove(action.card)
        counts = Counter(c.color for c in rest if c.color)
        if counts:
            diag["color_is_most_held"] = counts[action.color] == max(counts.values())
    return diag


def event_to_json(e: Event) -> dict:
    out = {"kind": e.kind, "player": e.player, "turn": e.turn}
    if e.top is not None:
        out["top"] = e.top.id
    if e.top_color is not None:
        out["top_color"] = e.top_color.value
    if e.card is not None:
        out["card"] = e.card.id
    if e.color is not None:
        out["color"] = e.color.value
    if e.cards:
        out["cards"] = [c.id for c in e.cards]
    if e.count:
        out["count"] = e.count
    if e.target >= 0:
        out["target"] = e.target
    if e.guilty is not None:
        out["guilty"] = e.guilty
    return out
