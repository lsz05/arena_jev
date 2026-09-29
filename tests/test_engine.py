import random
from collections import Counter

import pytest

from arena.players import HeuristicPlayer, RandomPlayer
from arena.uno.cards import COLORS, Card, Color, build_deck
from arena.uno.engine import (
    DRAWN_CARD, START_COLOR, TURN, ChooseColor, Draw, IllegalAction, Keep, Play, UnoGame,
)

R, Y, G, B = Color.RED, Color.YELLOW, Color.GREEN, Color.BLUE


def c(card_id: str) -> Card:
    return Card.from_id(card_id)


def stacked(hands: list[list[str]], start: str, rest: list[str] = ()) -> list[Card]:
    """A deck (top first) that deals `hands` (7 cards each), flips `start`, then yields `rest`."""
    n = len(hands)
    assert all(len(h) == 7 for h in hands)
    deck = [c(hands[p][r]) for r in range(7) for p in range(n)]
    return deck + [c(start)] + [c(x) for x in rest]


FILLER = ["yellow_1", "yellow_2", "yellow_3", "yellow_4", "yellow_5", "yellow_6", "yellow_7"]


def test_deck_composition():
    deck = build_deck()
    assert len(deck) == 108
    counts = Counter(deck)
    for color in COLORS:
        assert counts[Card(color, "0")] == 1
        for rank in [str(i) for i in range(1, 10)] + ["skip", "reverse", "draw_two"]:
            assert counts[Card(color, rank)] == 2
    assert counts[c("wild")] == 4 and counts[c("wild_draw_four")] == 4
    assert sum(card.points for card in deck) == 4 * (45 * 2 + 0) + 4 * 6 * 20 + 8 * 50


def test_card_ids_round_trip():
    for card in set(build_deck()):
        assert Card.from_id(card.id) == card


def test_deal_and_card_conservation():
    game = UnoGame(4, seed=1)
    assert all(len(h) == 7 for h in game.hands)
    rng = random.Random(0)
    bot = RandomPlayer()
    while not game.over:
        d = game.decision()
        game.step(bot.act(game.observation(d.player), d, rng).action)
        assert len(game.draw_pile) + len(game.discard) + sum(map(len, game.hands)) == 108


def test_playable_matches_color_or_rank():
    hands = [["red_5", "blue_7", "green_3", "wild", "yellow_9", "blue_skip", "red_0"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_5"))
    d = game.decision()
    assert d.kind == TURN and d.player == 0
    assert {x.id for x in d.playable} == {"red_5", "blue_7", "blue_skip", "wild"}


def test_wild_draw_four_only_without_current_color():
    hands = [["wild_draw_four", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_9"))
    assert {x.id for x in game.decision().playable} == {"wild_draw_four"}  # no blue in hand
    hands[0][1] = "blue_1"
    game = UnoGame(4, deck=stacked(hands, "blue_9"))
    assert "wild_draw_four" not in {x.id for x in game.decision().playable}


def test_illegal_action_rejected():
    hands = [["red_5", "blue_7", "green_3", "yellow_1", "yellow_9", "blue_skip", "red_0"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_5"))
    with pytest.raises(IllegalAction):
        game.step(Play(c("green_3")))
    with pytest.raises(IllegalAction):
        game.step(Play(c("wild"), R))  # not in hand


def test_skip_reverse_draw_two():
    hands = [["red_skip", "red_reverse", "red_draw_two", "red_1", "red_2", "red_3", "red_4"]] + [FILLER] * 3
    rest = ["green_1", "green_2", "green_3", "green_4"]
    game = UnoGame(4, deck=stacked(hands, "red_9", rest))
    game.step(Play(c("red_skip")))
    assert game.current == 2  # P1 skipped
    # Reverse from P2 would need red; check direction flip directly on P0's next turn instead
    game = UnoGame(4, deck=stacked(hands, "red_9", rest))
    game.step(Play(c("red_reverse")))
    assert game.direction == -1 and game.current == 3
    game = UnoGame(4, deck=stacked(hands, "red_9", rest))
    game.step(Play(c("red_draw_two")))
    assert game.current == 2 and len(game.hands[1]) == 9
    assert [x.id for x in game.hands[1][-2:]] == ["green_1", "green_2"]


def test_reverse_is_skip_with_two_players():
    hands = [["red_reverse", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6"], FILLER]
    game = UnoGame(2, deck=stacked(hands, "red_9"))
    game.step(Play(c("red_reverse")))
    assert game.current == 0


def test_wild_draw_four_effect_and_color():
    hands = [["wild_draw_four", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_9", ["green_1", "green_2", "green_3", "green_4"]))
    game.step(Play(c("wild_draw_four"), R))
    assert game.color == R and game.current == 2 and len(game.hands[1]) == 11


def test_draw_then_play_or_keep():
    hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "red_7"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_9", ["blue_3", "green_8"]))
    game.step(Draw())  # drawing is allowed although nothing else fits anyway
    d = game.decision()
    assert d.kind == DRAWN_CARD and d.player == 0 and d.playable == (c("blue_3"),)
    game.step(Keep())
    assert game.current == 1 and len(game.hands[0]) == 8
    # unplayable drawn card: the turn passes at once
    game = UnoGame(4, deck=stacked(hands, "blue_9", ["green_8"]))
    game.step(Draw())
    assert game.current == 1


def test_draw_allowed_with_playable_card():
    hands = [["blue_1", "red_2", "red_3", "red_4", "red_5", "red_6", "red_7"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_9", ["green_8"]))
    assert Draw() in game.decision().legal_actions()


def test_start_card_rules():
    hands = [FILLER] * 4
    game = UnoGame(4, deck=stacked(hands, "red_skip"))
    assert game.current == 1
    game = UnoGame(4, deck=stacked(hands, "red_draw_two", ["green_1", "green_2"]))
    assert game.current == 1 and len(game.hands[0]) == 9
    game = UnoGame(4, deck=stacked(hands, "red_reverse"))
    assert game.current == 3 and game.direction == -1
    game = UnoGame(4, deck=stacked(hands, "wild"))
    d = game.decision()
    assert d.kind == START_COLOR and d.player == 0
    game.step(ChooseColor(G))
    assert game.color == G and game.decision() == game.decision() and game.decision().kind == TURN
    assert game.current == 0
    game = UnoGame(4, deck=stacked(hands, "wild_draw_four", ["red_5"] * 10), seed=3)
    assert game.top_card.rank != "wild_draw_four"


def test_win_and_score():
    hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "red_7"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "red_9"))
    game.hands[0] = [c("red_draw_two")]
    game.step(Play(c("red_draw_two")))
    assert game.over and game.winner == 0
    r = game.result()
    assert r.hand_points[1] == sum(range(1, 8)) + sum(x.points for x in game.hands[1][7:])
    assert r.score == sum(r.hand_points)


def test_reshuffle_when_draw_pile_empty():
    hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "red_7"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "red_9", []))
    game.discard = [c("blue_1"), c("blue_2"), c("red_9")]  # pretend blue_1, blue_2 were played earlier
    assert not game.draw_pile
    game.step(Draw())
    assert len(game.hands[0]) == 8 and game.discard == [c("red_9")]


def test_hidden_information_in_observations():
    game = UnoGame(4, seed=5)
    rng = random.Random(1)
    bot = RandomPlayer()
    for _ in range(80):
        if game.over:
            break
        d = game.decision()
        game.step(bot.act(game.observation(d.player), d, rng).action)
    for p in range(4):
        for e in game.observation(p).events():
            if e.cards:
                assert e.player == p


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_random_games_terminate(n):
    rng = random.Random(n)
    bots = [RandomPlayer() if i % 2 else HeuristicPlayer() for i in range(n)]
    for seed in range(30):
        game = UnoGame(n, seed)
        while not game.over:
            d = game.decision()
            choice = bots[d.player].act(game.observation(d.player), d, rng)
            assert choice.action in d.legal_actions()
            game.step(choice.action)
        assert game.winner is not None
