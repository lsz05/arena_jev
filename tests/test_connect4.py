import json
import random

import pytest

from arena.connect4 import render
from arena.connect4.engine import COLS, ROWS, Connect4Game, IllegalMove
from arena.connect4.match import openings, play_game, show, summarize
from arena.connect4.players import HeuristicPlayer, RandomPlayer, SystemOnePlayer
from arena.mock_server import serve
from arena.systemone import SystemOneClient


def game_of(cols_1based: str) -> Connect4Game:
    return Connect4Game([int(c) - 1 for c in cols_1based])


@pytest.mark.parametrize("moves, winner", [
    ("1212121", 0),        # vertical in column 1
    ("1122334", 0),        # horizontal on row 1
    ("12234334544", 0),    # diagonal (1,1)-(4,4)
    ("7665545444", None),  # three in a line for both, no four yet
])
def test_wins(moves, winner):
    g = game_of(moves)
    assert g.winner == winner and g.over == (winner is not None)
    if winner is not None:
        assert len(g.line) >= 4


def four_in_a_row(g: Connect4Game) -> set:
    found = set()
    for c in range(COLS):
        for r in range(ROWS):
            for dc, dr in ((1, 0), (0, 1), (1, 1), (1, -1)):
                cells = [(c + k * dc, r + k * dr) for k in range(4)]
                if all(0 <= x < COLS and 0 <= y < ROWS for x, y in cells):
                    vals = {g.cell(x, y) for x, y in cells}
                    if len(vals) == 1 and None not in vals:
                        found |= vals
    return found


def test_engine_agrees_with_a_brute_force_scan():
    for seed in range(300):
        rng = random.Random(seed)
        g = Connect4Game()
        while not g.over:
            g.play(rng.choice(g.legal()))
        assert four_in_a_row(g) == (set() if g.winner is None else {g.winner})


def test_illegal_moves_and_full_column():
    g = game_of("111111")
    assert 0 not in g.legal()
    with pytest.raises(IllegalMove):
        g.play(0)
    with pytest.raises(IllegalMove):
        game_of("1212121").play(3)  # the game is over


def test_a_full_board_is_a_draw():
    # players who never complete four in a row when they can avoid it fill the board sooner or later
    for seed in range(300):
        rng = random.Random(seed)
        g = Connect4Game()
        while not g.over:
            legal = g.legal()
            quiet = [c for c in legal if not g.wins_with(c, g.to_move)]
            g.play(rng.choice(quiet or legal))
        if g.winner is None:
            assert g.ply == COLS * ROWS and g.over
            return
    pytest.fail("no drawn game found")


def test_render_is_from_the_movers_view_and_lists_legal_columns():
    g = game_of("4411111")  # 7 discs: player 1 (second) to move; column 1 holds 5 discs
    d = g.decision()
    state = render.render_state(g, d)
    assert "You are X" in state and "Moves so far, oldest first:" in state
    # the mover is player 1 (7 discs played): its own discs are X
    assert render.moves_text(g, d.player).startswith("O4 X4")
    q = render.question(g, d, "B", random.Random(0))
    assert set(q.options) == {str(c + 1) for c in d.legal}
    assert q.options["1"] == "Drop a disc in column 1: it lands on row 6."
    q2 = render.question(g, d, "B", random.Random(0))
    assert list(q.options) == list(q2.options)  # the shuffle is reproducible


def test_move_list_is_dropped_only_above_the_models_limit():
    g = game_of("1234567")  # alternating on row 1: nobody wins
    d = g.decision()
    q = render.question(g, d, "B")
    count = lambda s: len(s.split())  # noqa: E731
    full, kept, n = render.fit_state(g, d, q, 10_000, count)
    assert kept and "Moves so far" in full
    short, kept, n2 = render.fit_state(g, d, q, n - 1, count)
    assert not kept and "Moves so far" not in short and n2 < n


def test_heuristic_takes_wins_and_blocks():
    h = HeuristicPlayer()
    g = game_of("112233")  # X to move with 1,2,3 on row 1: wins in column 4
    assert h.act(g, g.decision(), random.Random(0)).col == 3
    g = game_of("1122335")  # O to move: X threatens column 4 -> block
    assert h.act(g, g.decision(), random.Random(0)).col == 3


def test_model_player_plays_and_logs(tmp_path):
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        model = SystemOnePlayer("mock", SystemOneClient(url), max_options=3)
        decisions = []
        rec = play_game([model, RandomPlayer()], (3,), 7, "g", on_decision=decisions.append)
        assert rec["opening"] == [4] and rec["moves"][0] == 4 and rec["plies"] == len(rec["moves"])
        mine = [d for d in decisions if d["player"] == "mock"]
        assert mine and all(d["action"] in d["legal"] for d in mine)
        # 7 legal columns with max_options 3: groups of 3, 3 and 1 (2 calls), then the 3 winners (1 call)
        seven = [d for d in mine if len(d["legal"]) == 7 and d.get("calls")]
        assert seven and all(len(d["calls"]) == 3 for d in seven)
        assert all(sorted(c["probabilities"]) == sorted(c["options"]) for d in mine for c in d.get("calls", []))
    finally:
        server.shutdown()


def test_match_files_summary_and_replay(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    games, decs = [], []
    for i, op in enumerate(openings(7)):
        for players in ((HeuristicPlayer(), RandomPlayer()), (RandomPlayer(), HeuristicPlayer())):
            games.append(play_game(list(players), op, i, f"G{i:02d}-{players[0].name}", on_decision=decs.append))
    (out / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in games))
    (out / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decs))
    s = summarize(out)
    assert s["games"] == 14 and s["players"]["heuristic"]["games"] == 14
    assert s["players"]["heuristic"]["took_immediate_win"] == 1.0
    text = show(out, games[0]["game"])
    assert "ply 1:" in text and "opening move" in text


def test_replay_frames_rebuild_the_requests(tmp_path):
    from arena.connect4 import replay
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        run = tmp_path / "20260101-000000_c4-mock-vs-heuristic"
        run.mkdir()
        decs, games = [], []
        for i, op in enumerate(openings(7)[:3]):
            players = [SystemOnePlayer("mock", SystemOneClient(url), max_options=3), HeuristicPlayer()]
            games.append(play_game(players, op, 100 + i, f"G{i:02d}-ab", on_decision=decs.append))
        (run / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in games))
        (run / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decs))
        (run / "config.json").write_text(json.dumps({"a": "mock", "b": "heuristic", "openings": 7, "template": "B"}))
        runs = replay.list_runs(tmp_path)
        assert [r["run"] for r in runs] == [run.name] and runs[0]["games"] == 3
        assert [g["game"] for g in replay.games(run)] == ["G00-ab", "G01-ab", "G02-ab"]
        for g in games:
            data = replay.frames(run, g["game"])
            moves, end = data["frames"][:-1], data["frames"][-1]
            assert [f["action"] for f in moves] == g["moves"] and end["winner"] == g["winner"]
            asked = [f for f in moves if f.get("calls")]
            assert asked and all(f["matches_log"] for f in asked)
            assert all("Choose the column" in f["request"]["questions"]["move"]["instructions"] for f in asked)
            assert all(f["opening"] == (f["ply"] < len(g["opening"])) for f in moves)
        with pytest.raises(KeyError):
            replay.frames(run, "nope")
    finally:
        server.shutdown()


def test_defend_hint_is_optional_and_replayed(tmp_path):
    from arena.connect4 import replay
    g = game_of("1122")
    d = g.decision()
    plain, hinted = render.question(g, d, "B"), render.question(g, d, "B", hint="defend")
    assert plain.instructions == render.INSTRUCTIONS and hinted.instructions.startswith(render.INSTRUCTIONS + " While you")
    with pytest.raises(ValueError):
        render.question(g, d, "B", hint="nope")
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        run = tmp_path / "20260101-000000_c4-mock-vs-heuristic-defend"
        run.mkdir()
        decs = []
        rec = play_game([SystemOnePlayer("mock", SystemOneClient(url), hint="defend"), HeuristicPlayer()], (3,), 5, "G00-ab",
                        on_decision=decs.append, hint="defend")
        assert rec["rules"] == {"template": "B", "hint": "defend"}
        (run / "games.jsonl").write_text(json.dumps(rec) + "\n")
        (run / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in decs))
        frames = replay.frames(run, "G00-ab")["frames"]
        asked = [f for f in frames if f.get("calls")]
        assert asked and all(f["matches_log"] and "block that column" in f["request"]["questions"]["move"]["instructions"]
                             for f in asked)
        assert all("threat_cols" in x["diag"] for x in decs)
    finally:
        server.shutdown()


def test_league_plays_every_pair_and_resumes(tmp_path):
    from arena.connect4.league import League
    from arena.connect4.replay import list_runs
    config = {"league": True, "openings": 7, "template": "B", "policy": "argmax", "hint": None, "pair_workers": 3,
              "seed": 1}
    out = tmp_path / "20260101-000000_c4-league"
    out.mkdir()
    (out / "games.jsonl").touch()
    (out / "decisions.jsonl").touch()
    (out / "config.json").write_text(json.dumps({**config, "players": ["h1", "h2", "r1", "r2"]}))
    players = {"h1": HeuristicPlayer("h1"), "h2": HeuristicPlayer("h2"), "r1": RandomPlayer("r1"), "r2": RandomPlayer("r2")}
    league = League(out, players, config)
    league.run()
    games = [json.loads(line) for line in open(out / "games.jsonl")]
    assert len(games) == 6 * 14 and len({g["game"] for g in games}) == len(games)
    pairs = {tuple(sorted(g["players"])) for g in games}
    assert len(pairs) == 6
    summary = json.loads((out / "summary.json").read_text())
    assert summary["games"] == 84 and all(p["games"] == 42 for p in summary["players"].values())
    assert summary["players"]["h1"]["ts_score"] > summary["players"]["r1"]["ts_score"]
    assert json.loads((out / "status.json").read_text())["state"] == "done"
    assert list_runs(tmp_path)[0]["games"] == 84
    # resume: drop the last pair and play it again
    last_pair = games[-1]["pair"]
    kept = [g for g in games if g["pair"] != last_pair]
    (out / "games.jsonl").write_text("".join(json.dumps(g) + "\n" for g in kept))
    again = League(out, players, config)
    again.restore()
    assert len(again.done) == 5 and again.games == 70
    again.run()
    assert sum(1 for _ in open(out / "games.jsonl")) == 84
