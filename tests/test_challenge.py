import json
import random

from arena.match import play_game, presented
from arena.mock_server import serve
from arena.players import HeuristicPlayer, RandomPlayer, SystemOnePlayer
from arena.replay import frames
from arena.systemone import SystemOneClient
from arena.uno import render
from arena.uno.cards import Card, Color
from arena.uno.engine import CHALLENGE, TURN, Accept, Challenge, Play, UnoGame
from tests.test_engine import FILLER, stacked

R, G = Color.RED, Color.GREEN
REST = [f"green_{i}" for i in range(1, 10)] + [f"yellow_{i}" for i in range(1, 10)]


def game_with(hand0: list[str], start: str = "blue_9") -> UnoGame:
    return UnoGame(4, deck=stacked([hand0] + [FILLER] * 3, start, REST), wd4_challenge=True)


def test_bluff_is_playable_only_with_challenge_rule():
    hand = ["wild_draw_four", "blue_1", "red_2", "red_3", "red_4", "red_5", "red_6"]
    assert "wild_draw_four" in {c.id for c in game_with(hand).decision().playable}
    classic = UnoGame(4, deck=stacked([hand] + [FILLER] * 3, "blue_9", REST))
    assert "wild_draw_four" not in {c.id for c in classic.decision().playable}


def test_guilty_challenge_punishes_the_bluffer_and_challenger_plays_on():
    g = game_with(["wild_draw_four", "blue_1", "red_2", "red_3", "red_4", "red_5", "red_6"])
    g.step(Play(Card.from_id("wild_draw_four"), R))
    d = g.decision()
    assert d.kind == CHALLENGE and d.player == 1
    g.step(Challenge())
    assert len(g.hands[0]) == 6 + 4 and len(g.hands[1]) == 7
    assert g.decision().kind == TURN and g.decision().player == 1 and g.color == R


def test_failed_challenge_costs_six_cards_and_the_turn():
    g = game_with(["wild_draw_four", "red_1", "red_2", "red_3", "red_4", "red_5", "red_6"])
    g.step(Play(Card.from_id("wild_draw_four"), R))
    g.step(Challenge())
    assert len(g.hands[1]) == 13 and len(g.hands[0]) == 6 and g.current == 2


def test_accept_draws_four_and_skips():
    g = game_with(["wild_draw_four", "blue_1", "red_2", "red_3", "red_4", "red_5", "red_6"])
    g.step(Play(Card.from_id("wild_draw_four"), R))
    g.step(Accept())
    assert len(g.hands[1]) == 11 and g.current == 2


def test_last_card_wild_draw_four_cannot_be_challenged():
    g = game_with(["wild_draw_four", "blue_1", "red_2", "red_3", "red_4", "red_5", "red_6"])
    g.hands[0] = [Card.from_id("wild_draw_four")]
    g.step(Play(Card.from_id("wild_draw_four"), R))
    assert g.over and g.winner == 0 and len(g.hands[1]) == 11


def test_challenged_hand_is_shown_to_the_challenger_only():
    g = game_with(["wild_draw_four", "blue_1", "red_2", "red_3", "red_4", "red_5", "red_6"])
    g.step(Play(Card.from_id("wild_draw_four"), R))
    g.step(Challenge())
    shown = {p: [e for e in g.observation(p).events() if e.kind == "challenge"][0].cards for p in range(4)}
    assert len(shown[1]) == 6 and not shown[0] and not shown[2] and not shown[3]
    state = render.render_state(g.observation(1), g.decision(), 5)
    assert "You challenged P0's Wild Draw Four and were right, so P0 drew 4 cards and you play now. P0's hand was: Red 2" in state
    g.step(g.decision().legal_actions()[-1])  # P1 finishes its turn (draws)
    later_p1 = render.render_history(g.observation(1), 5)
    later_p2 = render.render_history(g.observation(2), 5)
    assert "challenged P0's Wild Draw Four: guilty; P0's hand: Red 2" in later_p1
    assert "P1 | top Wild Draw Four (Red) | challenged P0's Wild Draw Four: guilty, P0 drew 4 cards" in later_p2
    assert "P0's hand" not in later_p2


def test_wild_draw_four_text_and_challenge_question():
    g = game_with(["wild_draw_four", "blue_1", "red_2", "red_3", "red_4", "red_5", "red_6"])
    d, obs = g.decision(), g.observation(0)
    text = render.move_question(obs, d, "B").options["wild_draw_four"]
    assert "you hold 1 blue card, so a challenge would succeed" in text
    g.step(Play(Card.from_id("wild_draw_four"), R))
    d, obs = g.decision(), g.observation(1)
    q = render.challenge_question(obs, d, "B", random.Random(0))
    assert set(q.options) == {"challenge", "accept"} and "previous color (Blue)" in q.instructions
    assert "P0 just played Wild Draw Four on you and named Red. The color before it was Blue." in render.render_state(obs, d, 1)


def test_heuristic_never_bluffs_and_random_games_finish():
    bot = HeuristicPlayer()
    rng = random.Random(0)
    for seed in range(40):
        g = UnoGame(4, seed, wd4_challenge=True)
        players = [bot, RandomPlayer(), bot, RandomPlayer()]
        while not g.over:
            d = g.decision()
            obs = presented(g, d.player, seed, True)
            a = players[d.player].act(obs, d, rng).action
            if players[d.player] is bot and isinstance(a, Play) and a.card.rank == "wild_draw_four":
                assert not any(c.color == g.color for c in g.hands[d.player])
            g.step(a)
            assert len(g.draw_pile) + len(g.discard) + sum(map(len, g.hands)) == 108
        assert g.winner is not None


def test_shuffled_hand_is_reproducible_and_complete():
    g = UnoGame(4, 3)
    a, b = presented(g, 0, 3, True), presented(g, 0, 3, True)
    assert a.hand == b.hand and sorted(a.hand, key=str) == sorted(g.observation(0).hand, key=str)


def test_replay_rebuilds_prompts_with_challenges_and_shuffled_hands():
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        model = SystemOnePlayer("mock", SystemOneClient(url), history_rounds=5)
        players = [model, RandomPlayer("r1"), RandomPlayer("r2"), RandomPlayer("r3")]
        kinds = set()
        for seed in range(12):
            decisions = []
            record = play_game(players, seed, game_id=f"g{seed}", on_decision=decisions.append,
                               wd4_challenge=True, shuffle_hand=True)
            record, decisions = json.loads(json.dumps(record)), json.loads(json.dumps(decisions))
            kinds |= {d["kind"] for d in decisions if d["player"] == "mock"}
            replay = frames(record, decisions, {"mock": {"type": "systemone", "history_rounds": 5}}, {})
            prompts = [f["prompt"] for f in replay["frames"] if "prompt" in f]
            assert prompts and all(p["matches_log"] for p in prompts)
        assert CHALLENGE in kinds  # the mock model met at least one Wild Draw Four
    finally:
        server.shutdown()
