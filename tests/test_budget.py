import random

from arena.players import SystemOnePlayer
from arena.systemone import SystemOneClient
from arena.mock_server import serve
from arena.tournament import _budget
from arena.uno import render
from arena.uno.engine import UnoGame
from tests.test_render_players import advance


def test_fit_state_drops_oldest_rounds_first():
    game = UnoGame(4, seed=11)
    advance(game, 40, 11)
    d = game.decision()
    obs = game.observation(d.player)
    q = render.move_question(obs, d, "B", random.Random(0))
    count = len  # characters as a stand-in tokenizer
    full, rounds, n = render.fit_state(obs, d, q, 5, 10**9, count)
    assert rounds == 5
    limit = count(render.request_text(render.render_state(obs, d, 2), q))
    state, rounds, n = render.fit_state(obs, d, q, 5, limit, count)
    assert rounds == 2 and n <= limit and state == render.render_state(obs, d, 2)
    assert state.split("Last round:")[1] == full.split("Last round:")[1]  # the most recent rounds are the ones kept
    _, rounds, _ = render.fit_state(obs, d, q, 5, 1, count)
    assert rounds == 0


def test_budget_is_the_smaller_of_model_limit_and_run_cap():
    assert _budget({}, {"max_prompt_tokens": None}) is None
    assert _budget({"max_input_tokens": 512}, {"max_prompt_tokens": None}) == 512
    assert _budget({"max_input_tokens": 32768}, {"max_prompt_tokens": 2000}) == 2000


def test_player_logs_history_rounds_kept():
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        tight = SystemOnePlayer("t", SystemOneClient(url), history_rounds=5, max_prompt_tokens=900, count_tokens=len)
        roomy = SystemOnePlayer("r", SystemOneClient(url), history_rounds=5, max_prompt_tokens=10**6, count_tokens=len)
        game = UnoGame(4, seed=11)
        advance(game, 40, 11)
        while not (game.decision().kind == "turn" and game.decision().playable):  # a real choice, not a forced draw
            advance(game, game.steps + 1, 11)
        d = game.decision()
        a = tight.act(game.observation(d.player), d, random.Random(0)).info["calls"][0]
        b = roomy.act(game.observation(d.player), d, random.Random(0)).info["calls"][0]
        assert a["history_rounds"] < b["history_rounds"] == 5 and a["prompt_tokens"] <= 900
    finally:
        server.shutdown()
