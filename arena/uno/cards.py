"""UNO cards: the classic 108-card deck."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Color(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"

    @property
    def label(self) -> str:
        return self.value.capitalize()


COLORS: tuple[Color, ...] = tuple(Color)

NUMBERS = tuple(str(n) for n in range(10))
SKIP, REVERSE, DRAW_TWO = "skip", "reverse", "draw_two"
WILD, WILD_DRAW_FOUR = "wild", "wild_draw_four"
ACTIONS = (SKIP, REVERSE, DRAW_TWO)
RANKS = NUMBERS + ACTIONS + (WILD, WILD_DRAW_FOUR)

_RANK_LABELS = {SKIP: "Skip", REVERSE: "Reverse", DRAW_TWO: "Draw Two", WILD: "Wild", WILD_DRAW_FOUR: "Wild Draw Four"}


@dataclass(frozen=True)
class Card:
    color: Color | None  # None for Wild and Wild Draw Four
    rank: str

    @property
    def is_wild(self) -> bool:
        return self.color is None

    @property
    def id(self) -> str:
        """Stable identifier, also used as the option name shown to models: "red_5", "blue_skip", "wild"."""
        return self.rank if self.is_wild else f"{self.color.value}_{self.rank}"

    @property
    def label(self) -> str:
        rank = _RANK_LABELS.get(self.rank, self.rank)
        return rank if self.is_wild else f"{self.color.label} {rank}"

    @property
    def points(self) -> int:
        """Official scoring: face value for numbers, 20 for Skip/Reverse/Draw Two, 50 for wilds."""
        if self.rank in NUMBERS:
            return int(self.rank)
        return 50 if self.is_wild else 20

    @classmethod
    def from_id(cls, card_id: str) -> Card:
        if card_id in (WILD, WILD_DRAW_FOUR):
            return cls(None, card_id)
        color, _, rank = card_id.partition("_")
        if rank not in RANKS:
            raise ValueError(f"unknown card id {card_id!r}")
        return cls(Color(color), rank)

    def __str__(self) -> str:
        return self.label


def build_deck() -> list[Card]:
    """The 108-card deck: per color one 0 and two each of 1-9, Skip, Reverse, Draw Two; four of each wild."""
    deck: list[Card] = []
    for color in COLORS:
        deck.append(Card(color, "0"))
        for rank in NUMBERS[1:] + ACTIONS:
            deck += [Card(color, rank)] * 2
    deck += [Card(None, WILD)] * 4 + [Card(None, WILD_DRAW_FOUR)] * 4
    return deck


def sort_key(card: Card) -> tuple[int, int]:
    """Order a hand by color (wilds last), then rank."""
    color = COLORS.index(card.color) if card.color else len(COLORS)
    return color, RANKS.index(card.rank)
