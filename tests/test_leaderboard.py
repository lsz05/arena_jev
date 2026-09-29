import json

from arena import site
from arena.leaderboard import build, connect4_results, trueskill_ratings, with_connect4


def game(seats, winner, points):
    return {"seats": seats, "winner_seat": winner, "hand_points": points, "aborted": False}


def test_trueskill_orders_by_results():
    # A always wins, B always keeps the fewest points among the losers, D the most
    games = [game(["A", "B", "C", "D"], 0, [0, 5, 20, 60])] * 30 + [game(["D", "C", "B", "A"], 3, [70, 25, 3, 0])] * 30
    ratings, opponents, skipped = trueskill_ratings(games)
    order = sorted(ratings, key=lambda n: -ratings[n].mu)
    assert order == ["A", "B", "C", "D"] and skipped == 0
    assert len(opponents["A"]) == 60 * 3


def test_equal_points_tie_and_duplicate_seats_skipped():
    ratings, _, skipped = trueskill_ratings([game(["A", "B", "C", "D"], 0, [0, 10, 10, 10])] * 20)
    # tied players sit at different places in TrueSkill's rank chain, so they match only approximately
    assert abs(ratings["B"].mu - ratings["C"].mu) < 0.05 and ratings["A"].mu > ratings["B"].mu
    _, _, skipped = trueskill_ratings([game(["A", "H", "H", "H"], 0, [0, 1, 2, 3])])
    assert skipped == 1


def test_build_without_run_uses_registry_and_jevbench(tmp_path):
    board = build(None, registry={"x": {"name": "x", "display": "X", "params": 123_456_789}})
    assert board["games"] == 0 and isinstance(board["rows"], list)


def c4_run(runs, name, players, league=True, state="running", summary=True):
    d = runs / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"league": True, "players": list(players)} if league else {"a": "x", "b": "y"}))
    status = {"state": state, "pairs_done": 1, "pairs": 3, "games": 14}
    (d / "status.json").write_text(json.dumps(status))
    if summary:
        (d / "summary.json").write_text(json.dumps({"games": 14, "players": players, "status": status}))
    return d


def c4_player(score, wins, draws, losses):
    return {"games": wins + draws + losses, "wins": wins, "draws": draws, "losses": losses, "score": score,
            "ts_mu": 26.0, "ts_sigma": 1.5, "ts_score": 21.5, "took_immediate_win": 0.5, "chances_to_win": 8,
            "blocked_immediate_threat": 0.25, "threats_to_block": 4, "fallbacks": 0, "avg_latency_ms": 120.0}


def test_newest_connect4_league_is_merged_into_every_row(tmp_path):
    c4_run(tmp_path, "20260101-000000_c4-league", {"a": c4_player(0.1, 1, 0, 9)}, state="done")
    c4_run(tmp_path, "20260201-000000_c4-league", {"a": c4_player(0.75, 10, 1, 3), "b": c4_player(0.25, 3, 1, 10)})
    c4_run(tmp_path, "20260301-000000_c4-a-vs-b", {"a": c4_player(1.0, 14, 0, 0)}, league=False)  # a match, not a league
    board = {"games": 0, "rows": [{"name": "a", "jb_acc": 0.5}, {"name": "c", "jb_acc": 0.4}]}
    merged = with_connect4(board, tmp_path)
    assert merged["c4"] == {"run": "20260201-000000_c4-league", "state": "running", "pairs_done": 1, "pairs": 3, "games": 14}
    a, c = merged["rows"]
    assert (a["c4_score"], a["c4_wins"], a["c4_draws"], a["c4_losses"], a["c4_games"]) == (0.75, 10, 1, 3, 14)
    assert (a["c4_ts_score"], a["c4_took_win"], a["c4_win_chances"], a["c4_blocked"], a["c4_threats"]) == (21.5, 0.5, 8, 0.25, 4)
    assert a["jb_acc"] == 0.5 and a["c4_p_block"] is None  # absent summary keys become None
    assert c["jb_acc"] == 0.4 and c["c4_score"] is None and c["c4_games"] is None
    assert "c4_score" not in board["rows"][0] and "c4" not in board  # the board passed in is left unchanged
    assert with_connect4(board, tmp_path / "none")["c4"] is None


def test_league_before_its_first_summary_reports_progress_only(tmp_path):
    c4_run(tmp_path, "20260201-000000_c4-league", {}, summary=False)
    progress, results = connect4_results(tmp_path)
    assert progress["state"] == "running" and progress["pairs"] == 3 and results == {}


def test_site_board_merges_connect4_with_and_without_uno_run(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    c4_run(runs, "20260201-000000_c4-league", {"x": c4_player(0.5, 5, 0, 5)})
    monkeypatch.setattr(site, "RUNS", runs)
    registry = {"x": {"name": "x", "display": "X", "params": 1}}
    monkeypatch.setattr(site.leaderboard, "load_registry", lambda path=None: registry)
    board = site.Site("latest").board()  # no *_swiss run under runs/: registry and JevBench only
    assert board["c4"]["run"] == "20260201-000000_c4-league"
    assert [(r["name"], r["c4_score"]) for r in board["rows"]] == [("x", 0.5)]
    uno = tmp_path / "uno"
    uno.mkdir()
    (uno / "games.jsonl").touch()
    (uno / "config.toml").write_text('[run]\nname = "uno"\n')
    (uno / "leaderboard.json").write_text(json.dumps({"games": 3, "in_progress": False, "rows": [{"name": "x", "ts_score": 20.0}]}))
    board = site.Site(str(uno)).board()
    (row,) = board["rows"]
    assert board["games"] == 3 and board["c4"]["state"] == "running"
    assert (row["ts_score"], row["c4_wins"], row["c4_ts_score"]) == (20.0, 5, 21.5)
