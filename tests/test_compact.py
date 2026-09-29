import random

from arena.match import play_game
from arena.players import RandomPlayer
from arena.uno import render
from arena.uno.engine import CHALLENGE, DRAWN_CARD, TURN


def _option_texts(games=150):
    texts = set()

    class Spy(RandomPlayer):
        def act(self, obs, decision, rng):
            if decision.kind in (TURN, DRAWN_CARD):
                texts.update(render.move_question(obs, decision, "B", random.Random(0)).options.values())
            if decision.kind == CHALLENGE:
                texts.update(render.challenge_question(obs, decision, "B", random.Random(0)).options.values())
            return super().act(obs, decision, rng)

    for s in range(games):
        play_game([Spy(f"r{i}") for i in range(4)], seed=s, game_id=str(s), wd4_challenge=True, shuffle_hand=True)
    return texts


def test_compaction_fits_a_short_cap_and_keeps_the_action():
    words = lambda t: len(t.split())  # noqa: E731 - a crude tokenizer is enough for the test
    for text in _option_texts():
        short = render.compact_option(text, lambda t: words(t) <= 26)
        assert words(short) <= 26, short
        action = short.split(".")[0].split(" (")[0]  # the action stays: "Play Red 5", "Draw 1 card", "Challenge it", ...
        assert action in text or action.replace("Play ", "Play the ", 1) + " you just drew" in text, (action, text)
        if "It is your last card" in text:
            assert "It is your last card: you win the game." in short
        if words(text) <= 26:
            assert short == text  # nothing changes when it already fits


def test_compact_question_reports_only_changed_options():
    q = render.Question("move", "Choose.", {"draw": "Draw 1 card.",
                                            "red_5": "Play Red 5 (matches Red). P3 (4 cards) plays next. You will have 6 cards left."})
    q2, changed = render.compact_question(q, lambda t: len(t) <= 40)
    assert list(q2.options) == ["draw", "red_5"] and set(changed) == {"red_5"}
    assert q2.options["draw"] == "Draw 1 card." and len(q2.options["red_5"]) <= 40
