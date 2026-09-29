"""Turn an engine decision into System One questions: a plain-text state plus one Choice question.

The state is rendered to text here instead of being sent as a JSON object, so every server
receives the same characters (servers serialize JSON states differently).

Knobs, identical for every player in a run:
- template: "A" names options by the card only; "B" (default) also states each move's effect
  and consequences; "C" is B plus the full rules text. Descriptions state facts, never advice.
- history_rounds: how many past rounds to include. A round starts at one of the player's own
  turns, so "last round" means everything since their previous turn.

Questions are asked one per request: first the move, then (only if a wild was chosen) the color.
With the Wild Draw Four challenge rule, the player hit by one gets a challenge question instead.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass

from .cards import COLORS, DRAW_TWO, NUMBERS, REVERSE, SKIP, WILD_DRAW_FOUR, Card, Color
from .engine import CHALLENGE, DRAWN_CARD, START_COLOR, TURN, Accept, Action, Challenge, Decision, Event, Observation

TEMPLATE_VERSION = "uno-v1"
TEMPLATES = ("A", "B", "C")
MAX_STATE_CHARS = 6000  # about 1.5k tokens; oldest history rounds are dropped beyond this

GOAL = (
    "Choose the move that gives you the best chance of winning this game of UNO. "
    "The first player to get rid of all their cards wins."
)

RULES = (
    "Rules: a card can be played if it matches the current color, or the number or symbol of the top card. "
    "Wild can always be played and lets you name the next color. Wild Draw Four also names the color, "
    "but can only be played when you hold no card of the current color. "
    "Skip: the next player loses their turn. Reverse: the play order reverses. "
    "Draw Two: the next player draws 2 cards and loses their turn. "
    "Wild Draw Four: the next player draws 4 cards and loses their turn. "
    "Instead of playing you may draw 1 card; if it can be played you may play it at once, otherwise your turn ends."
)
RULES_CHALLENGE = (
    "Rules: a card can be played if it matches the current color, or the number or symbol of the top card. "
    "Wild can always be played and lets you name the next color. Wild Draw Four also names the color; it is only "
    "allowed when you hold no card of the current color, but it can be played anyway and the next player may challenge it. "
    "A challenge succeeds if the player held a card of that color: that player draws 4 cards and the challenger plays on. "
    "Otherwise the challenger draws 6 cards and loses the turn. Without a challenge the next player draws 4 cards and loses the turn. "
    "Skip: the next player loses their turn. Reverse: the play order reverses. "
    "Draw Two: the next player draws 2 cards and loses their turn. "
    "Instead of playing you may draw 1 card; if it can be played you may play it at once, otherwise your turn ends."
)


@dataclass(frozen=True)
class Question:
    name: str  # "move" or "color"
    instructions: str
    options: dict[str, str]  # option name -> description, in presentation order

    def to_json(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


# ---- names -----------------------------------------------------------------------------------


def seat(i: int) -> str:
    return f"P{i}"


def who(i: int, you: int) -> str:
    return "You" if i == you else seat(i)


def top_label(card: Card, color: Color | None) -> str:
    if not card.is_wild:
        return card.label
    return f"{card.label} ({color.label})" if color else f"{card.label} (no color named yet)"


def cards_text(n: int) -> str:
    return "1 card" if n == 1 else f"{n} cards"


# ---- state -----------------------------------------------------------------------------------


def render_state(obs: Observation, decision: Decision, history_rounds: int = 1) -> str:
    you = obs.player
    order = [obs.seat_after(k) for k in range(obs.num_players)]
    lines = [
        f"You are {seat(you)} in a {obs.num_players}-player game of UNO.",
        "Turn order: " + " → ".join(f"{seat(s)} (you)" if s == you else seat(s) for s in order),
        f"Your hand ({cards_text(len(obs.hand))}): " + ", ".join(c.label for c in obs.hand),
    ]
    if decision.kind == START_COLOR:
        lines.append(f"Top card: {obs.top_card.label} (the first card; you name the starting color)")
    else:
        color = f"current color: {obs.color.label}" if obs.color else "no color named yet"
        lines.append(f"Top card: {obs.top_card.label} ({color})")
    lines.append(
        "Cards left: "
        + ", ".join(f"{seat(s)}{' (next)' if k == 1 else ''} {obs.hand_sizes[s]}" for k, s in enumerate(order[1:], 1))
    )
    lines.append(f"Draw pile: {cards_text(obs.draw_pile_size)}")
    if decision.kind == DRAWN_CARD:
        lines.append(f"You just drew {decision.playable[0].label}. It can be played now, or you can keep it.")
    for e in obs.events():  # a successful challenge in this turn: the challenger plays on, with what it learned
        if e.turn == obs.turn and e.kind == "challenge" and e.player == you:
            hand = f" {seat(e.target)}'s hand was: {', '.join(c.label for c in e.cards)}." if e.cards else ""
            lines.append(f"You challenged {seat(e.target)}'s Wild Draw Four and were right, so {seat(e.target)} drew "
                         f"4 cards and you play now.{hand}")
    if decision.kind == CHALLENGE:
        off, prev, named = last_wild_draw_four(obs)
        lines.append(f"{seat(off)} just played Wild Draw Four on you and named {named.label}. The color before it was "
                     f"{prev.label}. {seat(off)} now holds {cards_text(obs.hand_sizes[off])}.")

    head = "\n".join(lines)
    for rounds in range(history_rounds, -1, -1):
        history = render_history(obs, rounds)
        text = head + ("\n\n" + history if history else "")
        if len(text) <= MAX_STATE_CHARS:
            return text
    return head


def request_text(state: str, question: Question) -> str:
    """The text a token budget is measured on: state, instructions and every option with its description."""
    return "\n".join([state, question.instructions] + [f"{k}: {v}" for k, v in question.options.items()])


def fit_state(obs: Observation, decision: Decision, question: Question, max_rounds: int,
              max_tokens: int | None = None, count_tokens=None) -> tuple[str, int, int | None]:
    """The state with the most history (up to `max_rounds`) whose request fits `max_tokens`.

    Rounds are dropped oldest first, one at a time; with no history left the state is used as is.
    Returns (state, rounds kept, token count or None when there is no budget).
    """
    for rounds in range(max_rounds, -1, -1):
        state = render_state(obs, decision, rounds)
        if not max_tokens or count_tokens is None:
            return state, rounds, None
        n = count_tokens(request_text(state, question))
        if n <= max_tokens or rounds == 0:
            return state, rounds, n
    raise AssertionError("unreachable")


def render_history(obs: Observation, rounds: int) -> str:
    """The last `rounds` rounds, oldest first, one line per turn: player | top card at that moment | action.

    The turn in progress is left out; the state describes it (e.g. "You just drew ...").
    """
    if rounds <= 0:
        return ""
    by_turn: dict[int, list[Event]] = {}
    for e in obs.events():
        if e.kind != "reshuffle" and e.turn < obs.turn:
            by_turn.setdefault(e.turn, []).append(e)
    own = sorted({t for t, evs in by_turn.items() if evs[0].player == obs.player})
    from_start = len(own) < rounds
    starts = [0] + own if from_start else own[-rounds:]
    bounds = starts + [obs.turn]
    sections = []
    for i in range(len(starts)):
        k = len(starts) - i
        if from_start and i == 0:
            label = "Start of the game:"
        else:
            label = "Last round:" if k == 1 else f"{k} rounds ago:"
        body = [_turn_line(by_turn[t], obs.player) for t in range(bounds[i], bounds[i + 1]) if t in by_turn]
        if body:
            sections.append("\n".join([label] + body))
    if not sections:
        return ""
    return "Recent history, oldest first. Each line: player | top card at that moment | action.\n" + "\n".join(sections)


def _turn_line(events: list[Event], you: int) -> str:
    first = events[0]
    if first.kind == "start":
        return f"Start | first card | {first.card.label}"
    name = who(first.player, you)
    if first.kind == "skipped":
        context = f"skipped by {first.top.label}"
    else:
        context = f"top {top_label(first.top, first.top_color)}"
    parts: list[str] = []
    for i, e in enumerate(events):
        if e.kind == "color":
            parts.append(f"named {e.color.label} as the starting color")
        elif e.kind == "play":
            drew_first = i > 0 and events[i - 1].kind == "draw"
            played = f"played it: {e.card.label}" if drew_first and e.player != you else f"played {e.card.label}"
            if drew_first and e.player == you:
                played = "played it"
            parts.append(played + (f", named {e.color.label}" if e.color else ""))
        elif e.kind == "draw":
            if e.count == 0:
                parts.append("had no card to draw")
            elif e.cards:
                parts.append(f"drew {e.cards[0].label}")
            else:
                parts.append("drew 1 card")
        elif e.kind == "keep":
            parts.append("kept it")
        elif e.kind == "challenge":
            verdict = "guilty" if e.guilty else "not guilty"
            hand = f"; {who(e.target, you)}'s hand: {', '.join(c.label for c in e.cards)}" if e.cards else ""
            parts.append(f"challenged {who(e.target, you)}'s Wild Draw Four: {verdict}{hand}")
        elif e.kind == "penalty":
            drawn = f" ({', '.join(c.label for c in e.cards)})" if e.cards else ""
            parts.append(f"{who(e.player, you)} drew {cards_text(e.count)}{drawn}")
        elif e.kind == "skipped":
            if e.count == 0:
                parts.append("lost the turn")
            elif e.cards:
                parts.append(f"drew {cards_text(e.count)} ({', '.join(c.label for c in e.cards)})")
            else:
                parts.append(f"drew {cards_text(e.count)}")
    return f"{name} | {context} | {', '.join(parts)}"


# ---- questions -------------------------------------------------------------------------------


def move_question(obs: Observation, decision: Decision, template: str = "B", rng: random.Random | None = None) -> Question:
    """The first question of a TURN or DRAWN_CARD decision. Option names map back via `move_to_card`."""
    _check_template(template)
    options: dict[str, str] = {}
    if decision.kind == TURN:
        for card in decision.playable:
            options[card.id] = _play_text(obs, card, template)
        if decision.must_play and decision.playable:  # house rule: no drawing while a card can be played
            return Question("move", _instructions(template, obs.wd4_challenge, obs.must_play), _shuffled(options, rng))
        options["draw"] = (
            "Draw 1 card."
            if template == "A"
            else "Draw 1 card instead of playing. If it can be played, you may play it right away; otherwise your turn ends."
        )
    elif decision.kind == DRAWN_CARD:
        card = decision.playable[0]
        options["play"] = _play_text(obs, card, template, drawn=True)
        if not decision.must_play:  # house rule: a playable drawn card must be played
            nxt = _seat_with_cards(obs, obs.seat_after(1))
            options["keep"] = "Keep it." if template == "A" else f"Keep it. Your turn ends and {nxt} plays next."
    else:
        raise ValueError(f"no move question for {decision.kind}")
    return Question("move", _instructions(template, obs.wd4_challenge, obs.must_play), _shuffled(options, rng))


def color_question(obs: Observation, decision: Decision, card: Card | None, template: str = "B", rng: random.Random | None = None) -> Question:
    """Ask for a color: after choosing a wild (`card`), or at the start (START_COLOR, `card` None)."""
    _check_template(template)
    counts = Counter(c.color for c in obs.hand if c.color)
    options = {}
    for color in COLORS:
        if template == "A":
            options[color.value] = color.label
        else:
            n = counts[color]
            options[color.value] = f"Name {color.label}. You hold {n} {color.value} card{'' if n == 1 else 's'}."
    if decision.kind == START_COLOR:
        lead = "The first card is a Wild, so you name the starting color."
    else:
        lead = f"You are playing {card.label}."
    instructions = f"{lead} Name the color that gives you the best chance of winning."
    if template == "C":
        instructions += "\n" + rules_text(obs.wd4_challenge, obs.must_play)
    return Question("color", instructions, _shuffled(options, rng))


def challenge_question(obs: Observation, decision: Decision, template: str = "B", rng: random.Random | None = None) -> Question:
    """The question of a CHALLENGE decision: challenge the Wild Draw Four just played on you, or accept it."""
    _check_template(template)
    off, prev, _ = last_wild_draw_four(obs)
    if template == "A":
        options = {"challenge": "Challenge it.", "accept": "Accept it."}
    else:
        options = {
            "challenge": f"Challenge it. If {seat(off)} holds a {prev.value} card, {seat(off)} draws 4 cards and you "
                         f"play next; otherwise you draw 6 cards and lose your turn.",
            "accept": f"Accept it. You draw 4 cards and lose your turn; {_plays_next(obs, 1)}.",
        }
    instructions = (f"{seat(off)} played Wild Draw Four, which is only allowed when the player holds no card of the "
                    f"previous color ({prev.label}). Choose the option that gives you the best chance of winning this game of UNO.")
    if template == "C":
        instructions += "\n" + rules_text(True, obs.must_play)
    return Question("challenge", instructions, _shuffled(options, rng))


# Shortening steps for one option description, least important detail first. Used only for a model whose server caps
# the length of a single option, and only for an option above that cap (like history, which is trimmed only above a
# model's own input limit). The action itself and "It is your last card: you win the game." are never removed.
_COMPACT = [
    (re.compile(r" \(matches [^)]*\)"), ""),
    (re.compile(r" You will have \d+ cards? left\."), ""),
    (re.compile(r" (P\d) may challenge it: you hold \d+ \w+ cards?, so a challenge would succeed and you would draw 4 cards instead\."),
     r" If \1 challenges, you draw 4 instead."),
    (re.compile(r" (P\d) may challenge it: you hold no \w+ card, so a challenge would fail and P\d would draw 6 cards\."),
     r" If \1 challenges, \1 draws 6 instead."),
    (re.compile(r" cards and loses their turn"), " and loses their turn"),
    (re.compile(r"; (?:P\d \(\d+ cards?\) plays next|you play again)"), ""),
    (re.compile(r"^Play the (.+?) you just drew\."), r"Play \1."),
    (re.compile(r" You name the next color\."), ""),
]


def compact_option(text: str, fits) -> str:
    """Shorten `text` step by step until `fits(text)`; returns the shortest form if nothing fits."""
    for pattern, repl in _COMPACT:
        if fits(text):
            break
        text = pattern.sub(repl, text)
    return text


def compact_question(q: Question, fits) -> tuple[Question, dict[str, str]]:
    """`q` with every option that does not fit shortened; also {option: new text} for the ones changed."""
    changed = {k: compact_option(v, fits) for k, v in q.options.items() if not fits(v)}
    if not changed:
        return q, {}
    return Question(q.name, q.instructions, {k: changed.get(k, v) for k, v in q.options.items()}), changed


def challenge_action(option: str) -> Action:
    return Challenge() if option == "challenge" else Accept()


def last_wild_draw_four(obs: Observation) -> tuple[int, Color, Color]:
    """(who played it, the color before it, the color named) for the Wild Draw Four being challenged."""
    e = next(e for e in reversed(obs.log) if e.kind == "play" and e.card.rank == WILD_DRAW_FOUR)
    return e.player, e.top_color, e.color


def move_to_card(decision: Decision, option: str) -> Card | None:
    """The card an option of the move question plays, or None for draw / keep."""
    if option in ("draw", "keep"):
        return None
    if option == "play":
        return decision.playable[0]
    return Card.from_id(option)


def _instructions(template: str, challenge: bool = False, must_play: bool = False) -> str:
    return GOAL + ("\n" + rules_text(challenge, must_play) if template == "C" else "")


DRAW_RULE = "Instead of playing you may draw 1 card; if it can be played you may play it at once, otherwise your turn ends."
DRAW_RULE_MUST = ("If you can play a card you must play one. Only when you cannot, you draw 1 card; "
                  "if it can be played you must play it at once, otherwise your turn ends.")


def rules_text(challenge: bool = False, must_play: bool = False) -> str:
    text = RULES_CHALLENGE if challenge else RULES
    return text.replace(DRAW_RULE, DRAW_RULE_MUST) if must_play else text


def _play_text(obs: Observation, card: Card, template: str, drawn: bool = False) -> str:
    what = f"Play the {card.label} you just drew." if drawn else f"Play {card.label}."
    if template == "A":
        return what
    parts = [what]
    if not drawn and not card.is_wild:
        if card.color == obs.color:
            parts[0] = f"Play {card.label} (matches {obs.color.label})."
        else:
            parts[0] = f"Play {card.label} (matches the {card.label.split(' ', 1)[1]})."
    parts.append(_effect(obs, card))
    left = len(obs.hand) - 1
    parts.append("It is your last card: you win the game." if left == 0 else f"You will have {cards_text(left)} left.")
    return " ".join(parts)


def _effect(obs: Observation, card: Card) -> str:
    nxt = _seat_with_cards(obs, obs.seat_after(1))
    if card.rank == SKIP or (card.rank == REVERSE and obs.num_players == 2):
        return f"{nxt} loses their turn; {_plays_next(obs, 2)}."
    if card.rank == REVERSE:
        return f"The play order reverses; {_plays_next(obs, -1)}."
    if card.rank == DRAW_TWO:
        return f"{nxt} draws 2 cards and loses their turn; {_plays_next(obs, 2)}."
    if card.rank == WILD_DRAW_FOUR:
        text = f"You name the next color. {nxt} draws 4 cards and loses their turn; {_plays_next(obs, 2)}."
        if obs.wd4_challenge and len(obs.hand) > 1:
            k = sum(c.color == obs.color for c in obs.hand)
            nxt_seat = seat(obs.seat_after(1))
            if k:
                text += (f" {nxt_seat} may challenge it: you hold {k} {obs.color.value} card{'' if k == 1 else 's'}, "
                         f"so a challenge would succeed and you would draw 4 cards instead.")
            else:
                text += (f" {nxt_seat} may challenge it: you hold no {obs.color.value} card, so a challenge would fail "
                         f"and {nxt_seat} would draw 6 cards.")
        return text
    if card.is_wild:
        return f"You name the next color. {_upper_first(_plays_next(obs, 1))}."
    assert card.rank in NUMBERS
    return f"{_upper_first(_plays_next(obs, 1))}."


def _plays_next(obs: Observation, k: int) -> str:
    s = obs.seat_after(k)
    return "you play again" if s == obs.player else f"{_seat_with_cards(obs, s)} plays next"


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _seat_with_cards(obs: Observation, s: int) -> str:
    return f"{seat(s)} ({cards_text(obs.hand_sizes[s])})"


def _shuffled(options: dict[str, str], rng: random.Random | None) -> dict[str, str]:
    keys = list(options)
    if rng is not None:
        rng.shuffle(keys)
    return {k: options[k] for k in keys}


def _check_template(template: str) -> None:
    if template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
