"""Rebuild a logged game step by step, for the viewer.

The engine is deterministic given the deal seed and the actions, so replaying the logged actions
reproduces every intermediate state. For System One players the exact request is rebuilt as well:
the same per-decision rng reproduces the shuffled option order (checked against the logged order).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .uno import render
from .uno.cards import Card, Color, sort_key
from .match import presented
from .uno.engine import CHALLENGE, START_COLOR, Accept, Challenge, ChooseColor, Draw, Event, Keep, Play, UnoGame


def action_from_json(a: dict):
    if a["type"] == "play":
        return Play(Card.from_id(a["card"]), Color(a["color"]) if a["color"] else None)
    if a["type"] == "choose_color":
        return ChooseColor(Color(a["color"]))
    return {"draw": Draw, "keep": Keep, "challenge": Challenge, "accept": Accept}[a["type"]]()


def describe(e: Event, names: list[str]) -> str:
    who = names[e.player] if e.player >= 0 else ""
    if e.kind == "start":
        return f"First card: {e.card.label}"
    if e.kind == "color":
        return f"{who} names {e.color.label} as the starting color"
    if e.kind == "play":
        return f"{who} plays {e.card.label}" + (f" and names {e.color.label}" if e.color else "")
    if e.kind == "draw":
        return f"{who} draws " + (e.cards[0].label if e.cards else "nothing (no cards left)")
    if e.kind == "keep":
        return f"{who} keeps the drawn card"
    if e.kind == "skipped":
        drawn = f"draws {', '.join(c.label for c in e.cards)} and " if e.cards else ""
        return f"{who} {drawn}loses the turn"
    if e.kind == "challenge":
        verdict = "guilty" if e.guilty else "not guilty"
        return f"{who} challenges {names[e.target]}'s Wild Draw Four: {verdict} (held {e.color.label}: {'yes' if e.guilty else 'no'})"
    if e.kind == "penalty":
        return f"{who} draws {', '.join(c.label for c in e.cards)} as the challenge penalty"
    if e.kind == "reshuffle":
        return f"The discard pile is shuffled into a new draw pile ({e.count} cards)"
    return e.kind


def _snapshot(game: UnoGame) -> dict:
    return {
        "hands": [[c.id for c in sorted(h, key=sort_key)] for h in game.hands],
        "top": game.top_card.id,
        "color": game.color.value if game.color else None,
        "direction": game.direction,
        "draw_pile": len(game.draw_pile),
    }


def frames(game_record: dict, decisions: list[dict], player_specs: dict[str, dict] | None = None,
           run: dict | None = None) -> dict:
    """Replay one game. Returns {"meta": ..., "frames": [...]}, one frame per decision plus a final frame.

    player_specs / run (from the run's config) give each System One player's template and history,
    needed to rebuild its prompts; without them prompts are omitted.
    """
    names = game_record["seats"]
    run = run or {}
    player_specs = player_specs or {}
    rules = game_record.get("rules") or {}
    shuffle_hand = rules.get("shuffle_hand", False)
    game = UnoGame(len(names), game_record["seed"], wd4_challenge=rules.get("wd4_challenge", False),
                   must_play=rules.get("must_play", False))
    decisions = sorted(decisions, key=lambda d: d["step"])
    out = []
    intro = [describe(e, names) for e in game.log]
    logged = len(game.log)
    for d in decisions:
        if game.over:
            raise ValueError(f"{game_record['game']}: game ended before step {d['step']}")
        decision = game.decision()
        if decision.player != d["seat"] or decision.kind != d["kind"]:
            raise ValueError(f"{game_record['game']} step {d['step']}: replay diverged from the log")
        frame = {
            "step": d["step"],
            "turn": game.turn,
            "seat": decision.player,
            "player": names[decision.player],
            "kind": decision.kind,
            **_snapshot(game),
            "playable": [c.id for c in decision.playable],
            "action": d["action"],
            "diag": d.get("diag", {}),
        }
        for k in ("calls", "forced", "fallback", "error"):
            if k in d:
                frame[k] = d[k]
        spec = player_specs.get(names[decision.player])
        if spec is not None and spec.get("type", "systemone") == "systemone" and d.get("calls"):
            frame["prompt"] = _prompt(game, decision, d, spec, run, game_record["seed"], shuffle_hand)
        game.step(action_from_json(d["action"]))
        frame["events"] = [describe(e, names) for e in game.log[logged:]]
        logged = len(game.log)
        out.append(frame)
    result = game.result()
    if result.winner != game_record["winner_seat"]:
        raise ValueError(f"{game_record['game']}: replay winner {result.winner} != logged {game_record['winner_seat']}")
    final = {"step": game.steps, "turn": game.turn, "seat": None, "player": None, "kind": "end", **_snapshot(game),
             "events": [], "hand_points": list(result.hand_points)}
    out.append(final)
    meta = {k: game_record[k] for k in ("game", "seed", "rotation", "seats", "winner_seat", "winner", "score",
                                         "hand_points", "turns", "steps", "aborted") if k in game_record}
    meta["intro"] = intro
    return {"meta": meta, "frames": out}


def _prompt(game: UnoGame, decision, d: dict, spec: dict, run: dict, seed: int, shuffle_hand: bool = False) -> dict:
    """Rebuild the request(s) the System One player sent at this decision, with the history it kept."""
    template = spec.get("template", run.get("template", "B"))
    default_rounds = spec.get("history_rounds", run.get("history_rounds", 1))
    obs = presented(game, decision.player, seed, shuffle_hand)
    rng = random.Random(f"{seed}:{d['step']}")
    questions = []
    if decision.kind == START_COLOR:
        questions.append(render.color_question(obs, decision, None, template, rng))
    elif decision.kind == CHALLENGE:
        questions.append(render.challenge_question(obs, decision, template, rng))
    else:
        q = render.move_question(obs, decision, template, rng)
        questions.append(q)
        moves = [c["pick"] for c in d["calls"] if c["question"] == "move"]
        # the last call decides (knockout: the final); a single option is taken without asking (a forced play)
        final = moves[-1] if moves else (next(iter(q.options)) if len(q.options) == 1 else None)
        card = render.move_to_card(decision, final) if final else None
        if card is not None and card.is_wild:
            questions.append(render.color_question(obs, decision, card, template, rng))
    by_name = {q.name: q for q in questions}
    # One request per logged call. A question with more options than the model takes was asked as a knockout: groups
    # of the shuffled options in order, then the group winners; each call is rebuilt with exactly its own options.
    requests, matches = [], True
    for call in d["calls"]:
        q = by_name.get(call["question"])
        if q is None or any(o not in q.options for o in call["options"]):
            matches = False
            continue
        shortened = call.get("compacted") or {}  # options shortened for a model with a per-option length cap
        rounds = call.get("history_rounds", default_rounds)
        requests.append({"name": q.name, "state": render.render_state(obs, decision, rounds), "history_rounds": rounds,
                         "prompt_tokens": call.get("prompt_tokens"), "type": "choice", "instructions": q.instructions,
                         "criteria": {o: shortened.get(o, q.options[o]) for o in call["options"]}})
    knockout = False
    for name, q in by_name.items():
        calls = [c["options"] for c in d["calls"] if c["question"] == name]
        expected = list(q.options)
        if len(calls) == 1:
            matches = matches and calls[0] == expected
        elif calls:  # groups (a group of one option is not asked), then the winners
            knockout = True
            rest = iter(expected)
            in_order = all(o in rest for c in calls[:-1] for o in c)
            matches = matches and in_order and set(calls[-1]) <= set(expected)
    return {
        "state": requests[0]["state"] if requests else render.render_state(obs, decision, default_rounds),
        "requests": requests,
        "matches_log": matches,
        "knockout": knockout,
    }


# ---- run directory access ------------------------------------------------------------------


class RunData:
    """Lazy access to a run directory: the game index in memory, decisions read per game."""

    def __init__(self, run_dir: str | Path):
        import tomllib

        self.dir = Path(run_dir)
        self.games: dict[str, dict] = {}
        self.offsets: dict[str, list[int]] = {}
        self._games_pos = 0
        self._dec_pos = 0
        self.refresh()
        with open(self.dir / "config.toml", "rb") as f:
            self.config = tomllib.load(f)
        self.run = self.config.get("run", {})
        self.specs = {p["name"]: p for p in self.config.get("players", [])}

    def refresh(self) -> None:
        """Read what was appended since the last call (complete lines only), so a running tournament can be browsed."""
        path = self.dir / "games.jsonl"
        if path.exists():
            with open(path, "rb") as f:
                f.seek(self._games_pos)
                chunk = f.read()
            end = chunk.rfind(b"\n") + 1
            for line in chunk[:end].splitlines():
                if line.strip():
                    g = json.loads(line)
                    self.games[g["game"]] = g
            self._games_pos += end
        path = self.dir / "decisions.jsonl"
        if path.exists():
            with open(path, "rb") as f:
                f.seek(self._dec_pos)
                pos = self._dec_pos
                for line in f:
                    if not line.endswith(b"\n"):
                        break
                    start = line.find(b'"game": "') + 9
                    gid = line[start:line.find(b'"', start)].decode()
                    self.offsets.setdefault(gid, []).append(pos)
                    pos += len(line)
            self._dec_pos = pos

    def summary(self) -> dict:
        path = self.dir / "summary.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def servers(self) -> dict:
        path = self.dir / "servers.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def index(self) -> list[dict]:
        keys = ("game", "seed", "rotation", "seats", "winner_seat", "winner", "score", "hand_points", "turns", "steps", "aborted")
        return [{k: g.get(k) for k in keys} for g in self.games.values()]

    def decisions(self, game_id: str) -> list[dict]:
        out = []
        with open(self.dir / "decisions.jsonl", "rb") as f:
            for pos in self.offsets.get(game_id, []):
                f.seek(pos)
                out.append(json.loads(f.readline()))
        return out

    def replay(self, game_id: str) -> dict:
        return frames(self.games[game_id], self.decisions(game_id), self.specs, self.run)
