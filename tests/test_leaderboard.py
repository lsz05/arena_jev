from arena.leaderboard import build, trueskill_ratings


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
