"""House rule must_play: with a playable card a player must play; a playable drawn card must be played."""
import json
import random

from arena.match import play_game
from arena.mock_server import serve
from arena.players import HeuristicPlayer, RandomPlayer, SystemOnePlayer
from arena.replay import frames
from arena.systemone import SystemOneClient
from arena.uno import render
from arena.uno.engine import DRAWN_CARD, TURN, Draw, Keep, Play, UnoGame
from tests.test_engine import FILLER, c, stacked


def test_no_drawing_while_a_card_can_be_played():
    hands = [["red_3"] + FILLER[:6]] + [FILLER] * 3
    for must in (False, True):
        g = UnoGame(4, deck=stacked(hands, "red_9"), must_play=must)
        d = g.decision()
        assert d.kind == TURN and c("red_3") in d.playable
        assert (Draw() in d.legal_actions()) is not must
        q = render.move_question(g.observation(0), d, "B", random.Random(0))
        assert ("draw" in q.options) is not must


def test_draw_only_when_nothing_fits_and_play_the_drawn_card():
    hands = [["green_1", "green_2", "green_3", "green_4", "green_5", "green_6", "green_7"]] + [FILLER] * 3
    for must in (False, True):
        g = UnoGame(4, deck=stacked(hands, "blue_9", ["blue_3"]), must_play=must)
        d = g.decision()
        assert d.playable == () and d.legal_actions() == [Draw()]
        g.step(Draw())
        d = g.decision()
        assert d.kind == DRAWN_CARD and d.playable == (c("blue_3"),)
        assert (Keep() in d.legal_actions()) is not must and Play(c("blue_3")) in d.legal_actions()
        q = render.move_question(g.observation(0), d, "B", random.Random(0))
        assert set(q.options) == ({"play"} if must else {"play", "keep"})


def test_rules_text_and_no_stalled_games():
    assert "you must play one" in render.rules_text(True, True) and "you may draw" not in render.rules_text(True, True)
    assert "you may draw" in render.rules_text(True, False)
    for s in range(60):
        g = play_game([RandomPlayer(f"r{i}") for i in range(4)], s, game_id=str(s), wd4_challenge=True,
                      shuffle_hand=True, must_play=True, max_steps=1000)
        assert not g["aborted"] and g["rules"]["must_play"] is True


def test_replay_of_a_must_play_game():
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        model = SystemOnePlayer("mock", SystemOneClient(url), history_rounds=5)
        players = [model, HeuristicPlayer("h"), RandomPlayer("r"), HeuristicPlayer("h2")]
        for seed in range(4):
            decisions = []
            record = play_game(players, seed, game_id=f"g{seed}", on_decision=decisions.append, wd4_challenge=True,
                               shuffle_hand=True, must_play=True)
            record, decisions = json.loads(json.dumps(record)), json.loads(json.dumps(decisions))
            assert not any(d["action"]["type"] in ("draw", "keep") and d["playable"] for d in decisions)
            replay = frames(record, decisions, {"mock": {"type": "systemone", "history_rounds": 5}}, {"template": "B"})
            prompts = [f["prompt"] for f in replay["frames"] if "prompt" in f]
            assert prompts and all(p["matches_log"] for p in prompts)
            assert not any("draw" in r["criteria"] for p in prompts for r in p["requests"] if r["name"] == "move"
                           and len(r["criteria"]) > 1)
    finally:
        server.shutdown()
