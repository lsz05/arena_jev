"""UNO rules engine: classic deck, official rules, no house rules.

The engine owns every rule. At each step it exposes one Decision (whose turn it is and what
kind of choice they face); a player answers with one of `decision.legal_actions()` and
`step()` applies it, so an illegal move cannot happen.

Rule choices:
- Drawing, with `must_play=False` (default, official rule): always allowed, even with a playable
  card. A playable drawn card may be played at once (a DRAWN_CARD decision); otherwise the turn passes.
  With `must_play=True` (house rule): a player holding a playable card must play one, and draws only
  when nothing can be played; a playable drawn card must then be played at once. Without it, weak
  players that keep drawing can stall a game forever once every card is in someone's hand.
- Wild Draw Four, with `wd4_challenge=False` (default): only playable when the player holds no
  card of the current color, so there is nothing to challenge.
  With `wd4_challenge=True` (official rule): playable at any time, and the next player decides
  whether to challenge (a CHALLENGE decision). If the player who played it held a card of the
  previous color, that player draws 4 and the challenger plays on normally; otherwise the
  challenger draws 6 and loses the turn. Accepting means drawing 4 and losing the turn. The
  challenged hand is shown to the challenger only. A Wild Draw Four that empties the hand ends
  the game and cannot be challenged.
- No stacking, no 7-0, no jump-in. Calling UNO is automatic.
- The first card follows the official rules: a Wild Draw Four is shuffled back, a Wild lets
  the first player name the color, and Skip / Reverse / Draw Two take effect at once.
- With two players Reverse acts as Skip.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .cards import COLORS, DRAW_TWO, REVERSE, SKIP, WILD_DRAW_FOUR, Card, Color, build_deck, sort_key

HAND_SIZE = 7

# Decision kinds
TURN = "turn"                # play a card or draw
DRAWN_CARD = "drawn_card"    # play the card just drawn, or keep it
START_COLOR = "start_color"  # the first card is a Wild: name the starting color
CHALLENGE = "challenge"      # a Wild Draw Four was played on you: challenge it or accept it


@dataclass(frozen=True)
class Play:
    card: Card
    color: Color | None = None  # the color named when playing a wild


@dataclass(frozen=True)
class Draw:
    pass


@dataclass(frozen=True)
class Keep:
    pass


@dataclass(frozen=True)
class ChooseColor:
    color: Color


@dataclass(frozen=True)
class Challenge:
    pass


@dataclass(frozen=True)
class Accept:
    pass


Action = Play | Draw | Keep | ChooseColor | Challenge | Accept


def action_to_json(action: Action) -> dict:
    if isinstance(action, Play):
        return {"type": "play", "card": action.card.id, "color": action.color.value if action.color else None}
    if isinstance(action, ChooseColor):
        return {"type": "choose_color", "color": action.color.value}
    return {"type": {Draw: "draw", Keep: "keep", Challenge: "challenge", Accept: "accept"}[type(action)]}


@dataclass(frozen=True)
class Decision:
    player: int
    kind: str
    playable: tuple[Card, ...] = ()  # distinct playable cards; for DRAWN_CARD just the drawn card
    must_play: bool = False  # house rule: with a playable card, drawing (TURN) or keeping (DRAWN_CARD) is not allowed

    def legal_actions(self) -> list[Action]:
        if self.kind == START_COLOR:
            return [ChooseColor(c) for c in COLORS]
        if self.kind == CHALLENGE:
            return [Challenge(), Accept()]
        actions: list[Action] = []
        for card in self.playable:
            actions += [Play(card, c) for c in COLORS] if card.is_wild else [Play(card)]
        if not (self.must_play and self.playable):
            actions.append(Draw() if self.kind == TURN else Keep())
        return actions


@dataclass(frozen=True)
class Event:
    """One thing that happened. `turn` groups the events of one player's turn."""

    kind: str  # start | color | play | draw | keep | skipped | challenge | penalty | reshuffle
    player: int  # -1 for table events (start, reshuffle)
    turn: int
    top: Card | None = None  # top of the discard pile when the event happened
    top_color: Color | None = None  # current color when the event happened
    card: Card | None = None  # card played, or the starting card
    color: Color | None = None  # color named with a wild or at the start
    cards: tuple[Card, ...] = ()  # cards drawn, or the challenged hand (hidden from other players in observations)
    count: int = 0  # number of cards drawn
    target: int = -1  # challenge: the player who played the Wild Draw Four
    guilty: bool | None = None  # challenge: whether that player held a card of the previous color


@dataclass(frozen=True)
class Observation:
    """What one player can see when it is their decision."""

    player: int
    num_players: int
    hand: tuple[Card, ...]
    top_card: Card
    color: Color | None
    direction: int  # +1: seat order P0 -> P1 -> ...; -1: reversed
    hand_sizes: tuple[int, ...]
    draw_pile_size: int
    turn: int  # the turn in progress; events with this turn number belong to it
    log: tuple[Event, ...] = field(repr=False)  # raw log; use events() to respect hidden information
    wd4_challenge: bool = False  # rule: Wild Draw Four may be played at any time and challenged
    must_play: bool = False  # house rule: a playable card must be played (draw only when nothing can be played)

    def seat_after(self, k: int = 1) -> int:
        return (self.player + k * self.direction) % self.num_players

    def events(self) -> list[Event]:
        """The game log as this player may see it: other players' cards are hidden."""
        return [e if e.player == self.player or not e.cards else replace(e, cards=()) for e in self.log]


@dataclass(frozen=True)
class Result:
    winner: int | None  # None when the game hit the step limit
    hand_points: tuple[int, ...]
    score: int  # points the winner collects from the other hands
    turns: int
    steps: int


class GameOver(Exception):
    pass


class IllegalAction(Exception):
    pass


class UnoGame:
    def __init__(self, num_players: int = 4, seed: int = 0, *, deck: list[Card] | None = None, max_steps: int = 5000,
                 wd4_challenge: bool = False, must_play: bool = False):
        if not 2 <= num_players <= 10:
            raise ValueError("UNO needs 2 to 10 players")
        self.n = num_players
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.wd4_challenge = wd4_challenge
        self.must_play = must_play
        if deck is None:
            deck = build_deck()
            self.rng.shuffle(deck)
        self.draw_pile = list(reversed(deck))  # deck[0] is the top card; draw with pop()
        self.discard: list[Card] = []
        self.hands: list[list[Card]] = [[] for _ in range(num_players)]
        self.color: Color | None = None
        self.direction = 1
        self.current = num_players - 1  # the dealer; P0 (to the dealer's left) starts
        self.turn = 0
        self.steps = 0
        self.winner: int | None = None
        self.drawn_card: Card | None = None
        self.awaiting_start_color = False
        self.pending_challenge: dict | None = None  # {"offender", "prev_color", "guilty"} while a CHALLENGE is open
        self.log: list[Event] = []
        for _ in range(HAND_SIZE):
            for hand in self.hands:
                hand.append(self.draw_pile.pop())
        self.initial_hands = tuple(tuple(h) for h in self.hands)
        self._flip_start_card()

    # ---- public API --------------------------------------------------------------

    @property
    def over(self) -> bool:
        return self.winner is not None or self.steps >= self.max_steps

    @property
    def top_card(self) -> Card:
        return self.discard[-1]

    def decision(self) -> Decision:
        if self.over:
            raise GameOver
        p = self.current
        if self.awaiting_start_color:
            return Decision(p, START_COLOR)
        if self.pending_challenge is not None:
            return Decision(p, CHALLENGE)
        if self.drawn_card is not None:
            return Decision(p, DRAWN_CARD, (self.drawn_card,), self.must_play)
        return Decision(p, TURN, self.playable(self.hands[p]), self.must_play)

    def playable(self, hand: list[Card]) -> tuple[Card, ...]:
        return tuple(sorted({c for c in hand if self.can_play(c, hand)}, key=sort_key))

    def can_play(self, card: Card, hand: list[Card]) -> bool:
        if card.rank == WILD_DRAW_FOUR:
            return self.wd4_challenge or not self.holds_color(hand)
        if card.is_wild:
            return True
        return card.color == self.color or card.rank == self.top_card.rank

    def holds_color(self, hand: list[Card], color: Color | None = None) -> bool:
        color = self.color if color is None else color
        return any(c.color == color for c in hand)

    def observation(self, player: int) -> Observation:
        return Observation(
            player=player,
            num_players=self.n,
            hand=tuple(sorted(self.hands[player], key=sort_key)),
            top_card=self.top_card,
            color=self.color,
            direction=self.direction,
            hand_sizes=tuple(len(h) for h in self.hands),
            draw_pile_size=len(self.draw_pile),
            turn=self.turn,
            log=tuple(self.log),
            wd4_challenge=self.wd4_challenge,
            must_play=self.must_play,
        )

    def step(self, action: Action) -> None:
        decision = self.decision()
        if action not in decision.legal_actions():
            raise IllegalAction(f"{action} is not legal for {decision}")
        self.steps += 1
        p = decision.player

        if isinstance(action, ChooseColor):
            self._record("color", p, color=action.color)
            self.color = action.color
            self.awaiting_start_color = False  # the same player now takes a normal turn
        elif isinstance(action, (Challenge, Accept)):
            self._resolve_challenge(p, isinstance(action, Challenge))
        elif isinstance(action, Draw):
            drawn = self._draw(p, 1)
            self._record("draw", p, cards=tuple(drawn), count=len(drawn))
            if drawn and self.can_play(drawn[0], self.hands[p]):
                self.drawn_card = drawn[0]
            else:
                self._pass_turn()
        elif isinstance(action, Keep):
            self._record("keep", p)
            self.drawn_card = None
            self._pass_turn()
        else:
            self._play(p, action)

    def result(self) -> Result:
        points = tuple(sum(c.points for c in h) for h in self.hands)
        score = sum(points) if self.winner is not None else 0
        return Result(self.winner, points, score, self.turn, self.steps)

    # ---- internals -----------------------------------------------------------------

    def _play(self, p: int, action: Play) -> None:
        card = action.card
        prev_color = self.color
        self._record("play", p, card=card, color=action.color)
        self.hands[p].remove(card)
        self.discard.append(card)
        self.color = action.color if card.is_wild else card.color
        self.drawn_card = None
        if card.rank == SKIP:
            self._skip_next()
        elif card.rank == REVERSE:
            if self.n == 2:
                self._skip_next()
            else:
                self.direction = -self.direction
                self._pass_turn()
        elif card.rank == DRAW_TWO:
            self._skip_next(draw=2)
        elif card.rank == WILD_DRAW_FOUR:
            if self.wd4_challenge and self.hands[p]:
                guilty = self.holds_color(self.hands[p], prev_color)
                self.pending_challenge = {"offender": p, "prev_color": prev_color, "guilty": guilty}
                self._begin_turn(self._next_seat())  # the next player's turn opens with the CHALLENGE decision
            else:
                self._skip_next(draw=4)
        else:
            self._pass_turn()
        if not self.hands[p]:
            self.winner = p  # after the effects, so a final Draw Two / Wild Draw Four still counts

    def _resolve_challenge(self, p: int, challenged: bool) -> None:
        info, self.pending_challenge = self.pending_challenge, None
        offender = info["offender"]
        if not challenged:
            drawn = self._draw(p, 4)
            self._record("skipped", p, cards=tuple(drawn), count=len(drawn))
            self._pass_turn()
            return
        # the challenger sees the challenged hand: stored on the challenger's event, hidden from everyone else
        self._record("challenge", p, cards=tuple(sorted(self.hands[offender], key=sort_key)), target=offender,
                     guilty=info["guilty"], color=info["prev_color"])
        if info["guilty"]:
            drawn = self._draw(offender, 4)
            self._record("penalty", offender, cards=tuple(drawn), count=len(drawn))
            # the challenger now takes a normal turn
        else:
            drawn = self._draw(p, 6)
            self._record("skipped", p, cards=tuple(drawn), count=len(drawn))
            self._pass_turn()

    def _flip_start_card(self) -> None:
        while True:
            card = self.draw_pile.pop()
            if card.rank != WILD_DRAW_FOUR:
                break
            self.draw_pile.append(card)
            self.rng.shuffle(self.draw_pile)
        self.discard.append(card)
        self.color = card.color
        self._record("start", -1, card=card)
        if card.is_wild:
            self._pass_turn()
            self.awaiting_start_color = True
        elif card.rank == SKIP:
            self._skip_next()
        elif card.rank == DRAW_TWO:
            self._skip_next(draw=2)
        elif card.rank == REVERSE:
            self.direction = -1
            self._begin_turn(self.n - 1)  # the dealer plays first, then play goes the other way
        else:
            self._pass_turn()

    def _begin_turn(self, player: int) -> None:
        self.current = player
        self.turn += 1

    def _next_seat(self) -> int:
        return (self.current + self.direction) % self.n

    def _pass_turn(self) -> None:
        self._begin_turn(self._next_seat())

    def _skip_next(self, draw: int = 0) -> None:
        """The next player draws `draw` cards and loses their turn."""
        self._begin_turn(self._next_seat())
        victim = self.current
        drawn = self._draw(victim, draw)
        self._record("skipped", victim, cards=tuple(drawn), count=len(drawn))
        self._pass_turn()

    def _draw(self, player: int, k: int) -> list[Card]:
        drawn: list[Card] = []
        for _ in range(k):
            if not self.draw_pile:
                self._reshuffle()
                if not self.draw_pile:
                    break  # every card is in someone's hand
            drawn.append(self.draw_pile.pop())
        self.hands[player].extend(drawn)
        return drawn

    def _reshuffle(self) -> None:
        if len(self.discard) <= 1:
            return
        top = self.discard.pop()
        self.draw_pile, self.discard = self.discard, [top]
        self.rng.shuffle(self.draw_pile)
        self._record("reshuffle", -1, count=len(self.draw_pile))

    def _record(self, kind: str, player: int, **fields) -> None:
        top = self.discard[-1] if self.discard else None
        self.log.append(Event(kind, player, self.turn, top=top, top_color=self.color, **fields))
