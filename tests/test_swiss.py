import json

from arena.mock_server import serve
from arena.players import SystemOnePlayer
from arena.swiss import Tournament, load_config
from arena.systemone import SystemOneClient


def test_swiss_tournament_reaches_target_and_writes_files(tmp_path):
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        cfg_path = tmp_path / "t.toml"
        cfg_path.write_text('[run]\nname = "t"\ntarget_games = 8\nseed = 1\n')
        cfg = load_config(cfg_path)
        players = {f"m{i}": SystemOnePlayer(f"m{i}", SystemOneClient(url), history_rounds=5) for i in range(6)}
        out = tmp_path / "run"
        out.mkdir()
        t = Tournament(cfg, out, players)
        t.run_all()
        games = [json.loads(line) for line in open(out / "games.jsonl")]
        counts = {m: sum(m in g["seats"] for g in games) for m in players}
        assert all(c >= 8 for c in counts.values())
        assert all(g["rules"] == {"wd4_challenge": True, "shuffle_hand": True, "max_steps": 1000, "must_play": False} for g in games)
        assert all(len(set(g["seats"])) == 4 for g in games)
        status = json.loads((out / "status.json").read_text())
        board = json.loads((out / "leaderboard.json").read_text())
        assert status["state"] == "done" and board["games"] == len(games)
        rows = {r["name"]: r for r in board["rows"]}
        assert set(players) <= set(rows) and all(rows[m]["ts_mu"] is not None for m in players)
        assert sum(1 for _ in open(out / "decisions.jsonl")) > 0
    finally:
        server.shutdown()


def test_swiss_resume_continues_and_hold_keeps_a_model_out(tmp_path):
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        cfg_path = tmp_path / "t.toml"
        cfg_path.write_text('[run]\nname = "t"\ntarget_games = 4\nseed = 1\n')
        cfg = load_config(cfg_path)
        mk = lambda: {f"m{i}": SystemOnePlayer(f"m{i}", SystemOneClient(url), history_rounds=5) for i in range(5)}  # noqa: E731
        out = tmp_path / "run"
        out.mkdir()
        first = Tournament(cfg, out, mk())
        first.run_all()
        before = {m: first.ratings[m].mu for m in first.players}
        # an orphan decision of a table that never finished must not be counted, and its id must not be reused
        with open(out / "decisions.jsonl", "a") as f:
            f.write(json.dumps({"game": "T00099-R0", "step": 0, "player": "m0", "kind": "turn", "calls": []}) + "\n")
        (out / "hold.txt").write_text("m4\n")
        cfg["run"]["target_games"] = 12
        second = Tournament(cfg, out, mk())
        second.restore()
        assert {m: second.ratings[m].mu for m in second.players} == before
        assert second.table_no == 99 and second.agg.games == first.agg.games
        held_games = second.games_played["m4"]
        (out / "hold.txt").write_text("")  # released again after a while, so the run can finish
        second.cfg["run"]["target_games"] = 12
        second.run_all()
        games = [json.loads(line) for line in open(out / "games.jsonl")]
        assert len({g["game"] for g in games}) == len(games)
        assert all(int(g["game"][1:6]) > 99 for g in games[first.agg.games:])
        assert all(sum(m in g["seats"] for g in games) >= 12 for m in second.players)
        assert held_games < 12
    finally:
        server.shutdown()


def test_hold_file_is_read(tmp_path):
    cfg_path = tmp_path / "t.toml"
    cfg_path.write_text('[run]\nname = "t"\n')
    out = tmp_path / "run"
    out.mkdir()
    t = Tournament(load_config(cfg_path), out, {})
    assert t.held() == set()
    (out / "hold.txt").write_text("a\nb\n")
    assert t.held() == {"a", "b"}


def test_max_lead_keeps_counts_even(tmp_path):
    cfg_path = tmp_path / "t.toml"
    cfg_path.write_text('[run]\nname = "t"\nmax_lead = 8\n')
    out = tmp_path / "run"
    out.mkdir()
    t = Tournament(load_config(cfg_path), out, {f"m{i}": None for i in range(9)})
    for i in range(9):
        t.games_played[f"m{i}"] = 20
    t.games_played["m0"] = 4
    t.busy = {"m0"}  # the model behind is still playing
    assert t.pick_table() is None  # everyone free is 16 ahead of m0
    t.games_played["m0"] = 14
    assert t.pick_table() is not None  # 6 ahead: allowed
