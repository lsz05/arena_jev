import random

from arena.mock_server import serve
from arena.players import HeuristicPlayer, SystemOnePlayer
from arena.systemone import SystemOneClient
from arena.uno import render
from arena.uno.cards import Card
from arena.uno.engine import START_COLOR, TURN, Play, UnoGame
from tests.test_engine import FILLER, stacked


def advance(game: UnoGame, steps: int, seed: int = 0) -> None:
    bot = HeuristicPlayer()
    while not game.over and game.steps < steps:
        d = game.decision()
        game.step(bot.act(game.observation(d.player), d, random.Random(seed + game.steps)).action)


def test_move_options_match_legal_moves():
    for seed in range(20):
        game = UnoGame(4, seed)
        advance(game, 25, seed)
        if game.over:
            continue
        d = game.decision()
        if d.kind == START_COLOR:
            continue
        q = render.move_question(game.observation(d.player), d, "B", random.Random(0))
        expected = {c.id for c in d.playable} | {"draw"} if d.kind == TURN else {"play", "keep"}
        assert set(q.options) == expected


def test_shuffle_is_deterministic_and_changes_order():
    hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "wild"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "red_9"))
    d, obs = game.decision(), game.observation(0)
    orders = {tuple(render.move_question(obs, d, "B", random.Random(s)).options) for s in range(20)}
    assert len(orders) > 1
    a = render.move_question(obs, d, "B", random.Random(7)).options
    b = render.move_question(obs, d, "B", random.Random(7)).options
    assert list(a) == list(b)


def test_templates_differ_and_describe_effects():
    hands = [["red_draw_two", "red_2", "red_3", "red_4", "red_5", "red_6", "wild"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "red_9"))
    d, obs = game.decision(), game.observation(0)
    a = render.move_question(obs, d, "A").options
    b = render.move_question(obs, d, "B").options
    cq = render.move_question(obs, d, "C")
    assert a["red_draw_two"] == "Play Red Draw Two."
    assert "P1 (7 cards) draws 2 cards and loses their turn; P2 (7 cards) plays next." in b["red_draw_two"]
    assert "Rules:" in cq.instructions
    assert "win" not in b["red_2"].lower().replace("wins", "")  # no strategic advice


def test_history_rounds_anchor_on_own_turns():
    game = UnoGame(4, seed=11)
    advance(game, 40, 11)
    d = game.decision()
    obs = game.observation(d.player)
    h1 = render.render_history(obs, 1)
    h3 = render.render_history(obs, 3)
    assert h1.count("\n") < h3.count("\n")
    assert "Last round:" in h1 and "3 rounds ago:" in h3
    last = h1.split("Last round:\n", 1)[1].splitlines()
    assert last[0].startswith("You |")
    assert render.render_history(obs, 0) == ""


def test_state_hides_other_players_draws():
    game = UnoGame(4, seed=2)
    advance(game, 60, 2)
    for p in range(4):
        text = render.render_history(game.observation(p), 20)
        for line in text.splitlines():
            if " | " in line and not line.startswith("You") and "drew 1 card" not in line:
                assert "drew " not in line or "cards" in line.split("drew ", 1)[1].split(",")[0] or "(" not in line


def test_systemone_player_with_mock_server():
    server = serve(port=0, background=True)
    port = server.server_address[1]
    try:
        player = SystemOnePlayer("mock", SystemOneClient(f"http://127.0.0.1:{port}"), history_rounds=3)
        game = UnoGame(4, seed=4)
        players = [player, HeuristicPlayer(), HeuristicPlayer(), HeuristicPlayer()]
        wild_seen = False
        while not game.over:
            d = game.decision()
            choice = players[d.player].act(game.observation(d.player), d, random.Random(game.steps))
            assert choice.action in d.legal_actions()
            if d.player == 0:
                assert not choice.info.get("fallback"), choice.info
                calls = choice.info["calls"]
                assert calls and calls[0]["pick"] in calls[0]["options"]
                if isinstance(choice.action, Play) and choice.action.card.is_wild:
                    wild_seen = True
                    assert [c["question"] for c in calls] == ["move", "color"]
            game.step(choice.action)
        del wild_seen
    finally:
        server.shutdown()


def test_systemone_player_falls_back_on_errors():
    player = SystemOnePlayer("down", SystemOneClient("http://127.0.0.1:9", retries=0, timeout=1))
    game = UnoGame(4, seed=1)
    d = game.decision()
    choice = player.act(game.observation(d.player), d, random.Random(0))
    assert choice.info["fallback"] and choice.action in d.legal_actions()


def test_knockout_for_small_option_caps():
    server = serve(port=0, background=True)
    port = server.server_address[1]
    try:
        player = SystemOnePlayer("capped", SystemOneClient(f"http://127.0.0.1:{port}"), max_options=2)
        hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "red_7"]] + [FILLER] * 3
        game = UnoGame(4, deck=stacked(hands, "red_9"))
        d = game.decision()
        choice = player.act(game.observation(0), d, random.Random(0))
        assert not choice.info.get("fallback")
        assert all(len(call["options"]) <= 2 for call in choice.info["calls"])
        assert len(choice.info["calls"]) > 1
    finally:
        server.shutdown()


def test_card_from_move_option():
    hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "wild"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "red_9"))
    d = game.decision()
    assert render.move_to_card(d, "wild") == Card.from_id("wild")
    assert render.move_to_card(d, "draw") is None


def test_single_option_is_not_sent():
    player = SystemOnePlayer("down", SystemOneClient("http://127.0.0.1:9", retries=0, timeout=1))
    hands = [["red_1", "red_2", "red_3", "red_4", "red_5", "red_6", "red_7"]] + [FILLER] * 3
    game = UnoGame(4, deck=stacked(hands, "blue_9"))
    d = game.decision()
    assert d.playable == ()
    choice = player.act(game.observation(0), d, random.Random(0))
    assert choice.info.get("forced") and not choice.info.get("fallback") and choice.info["calls"] == []
