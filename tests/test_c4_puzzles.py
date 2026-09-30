import json
import random

from arena.connect4.engine import Connect4Game
from arena.connect4.players import SystemOnePlayer
from arena.connect4.puzzle_run import ask, baselines, load_puzzles, orders, score
from arena.connect4.puzzles import canonical, classify
from arena.mock_server import serve
from arena.systemone import SystemOneClient


def game(moves: str) -> Connect4Game:
    return Connect4Game([int(c) - 1 for c in moves])


def test_known_positions_are_classified():
    assert classify(game("2324261452672275654"))[:2] == ("win", 5)
    kind, col, detail = classify(game("32461175135152"))
    assert (kind, col) == ("block", 3) and detail["direction"] == "horizontal"
    for item in load_puzzles():  # the stored fork and solver puzzles classify again the same way
        if item["family"] in ("fork", "solver") and item["id"].endswith("0"):
            kind, col, detail = classify(game(item["moves"]))
            assert (kind, str(col + 1)) == (item["family"], item["expected"])
            if kind == "fork":
                assert all(int(item["expected"]) in cols or cols for cols in detail["replies"].values())


def test_mirrors_share_a_key():
    g = game("3246117513")
    mirrored = Connect4Game([6 - m for m in g.moves])
    assert canonical(g) == canonical(mirrored)


def test_orders_put_the_answer_in_different_places():
    for item in load_puzzles()[::37]:
        places = [names.index(item["expected"]) for names in orders(item, 3)]
        assert all(sorted(names) == sorted(item["labels"]) for names in orders(item, 3))
        if len(item["labels"]) >= 3:
            assert len(set(places)) == 3


def test_the_puzzle_set():
    items = load_puzzles()
    assert len(items) == 1000 and len({i["id"] for i in items}) == 1000
    assert len({canonical(game(i["moves"])) for i in items}) == 1000
    for t in ("win", "block", "fork", "solver"):
        cols = [int(i["expected"]) for i in items if i["family"] == t]
        assert len(cols) == 250 and max(cols.count(c) for c in range(1, 8)) - min(cols.count(c) for c in range(1, 8)) <= 2


def test_ask_and_score_with_a_mock_server():
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        full = SystemOnePlayer("mock", SystemOneClient(url))
        capped = SystemOnePlayer("mock", SystemOneClient(url), max_options=3)
        items = load_puzzles()[::100]
        rows = []
        for it in items:
            for k, names in enumerate(orders(it)):
                r = ask(full, capped, it, names, random.Random(0))
                assert r["pick"] in names and r["probs"] and abs(sum(r["probs"].values()) - 1) < 1e-6
                rows.append({"item": it["id"], "family": it["family"], "order": k, "options": names, "error": None, **r})
        s = score(rows, {i["id"]: i for i in items})
        assert s["answers"] == 3 * len(items) and 0 <= s["accuracy"] <= 1 and s["items"] == len(items)
        assert s["consistency"] is not None and s["brier"] is not None
    finally:
        server.shutdown()


def test_baselines():
    b = baselines(load_puzzles())
    # chance is 1 / (legal columns): about 0.25 overall, since late positions have few columns left
    assert 0.1 < b["random"]["accuracy"] < 0.35 and b["centre"]["accuracy"] < 0.4
    assert b["heuristic"]["by_type"]["win"]["accuracy"] == 1.0 and b["heuristic"]["by_type"]["block"]["accuracy"] == 1.0


def test_site_data_from_a_run(tmp_path):
    from arena.connect4 import puzzle_site
    from arena.connect4.puzzle_run import write_summary
    items = load_puzzles()[:20]
    run = tmp_path / "20260101-000000_c4-puzzles"
    (run / "m1").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"orders": 3}))
    rows = []
    for it in items:
        for k, names in enumerate(orders(it)):
            pick = it["expected"] if k < 2 else names[0]
            probs = {n: (0.7 if n == pick else 0.3 / (len(names) - 1)) for n in names}
            rows.append({"item": it["id"], "family": it["family"], "order": k, "options": names, "pick": pick,
                         "correct": pick == it["expected"], "probs": probs, "p_correct": probs[it["expected"]], "calls": 1,
                         "knockout": False, "with_moves": True, "latency_ms": 1.0, "error": None})
    (run / "m1" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_summary(run, items, final=True)
    s = puzzle_site.summary(tmp_path)
    assert s["run"] == run.name and s["state"] == "done" and [m["name"] for m in s["models"]] == ["m1"]
    assert 0.66 <= s["models"][0]["accuracy"] <= 1 and s["models"][0]["win_accuracy"] is not None
    listed = {i["id"]: i for i in puzzle_site.items(tmp_path)["items"]}
    assert listed[items[0]["id"]]["answers"] == 3
    d = puzzle_site.item(items[0]["id"], tmp_path)
    assert len(d["models"][0]["orders"]) == 3 and d["answers"] == 3 and sum(d["picks"].values()) == 3
