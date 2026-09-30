import json

from arena import benchdata
from arena.site import Site


def test_items_and_answers_are_joined(tmp_path):
    items = tmp_path / "items"
    items.mkdir()
    rows = [
        {"id": "easy-a-0", "family": "a", "group": None, "split": "public", "state": "S", "labels": ["x", "y"], "expected": "y",
         "question": {"type": "choice", "instructions": "Pick.", "criteria": {"x": "X it.", "y": "Y it."}}, "provenance": {"license": "MIT"}},
        {"id": "hard-b-0", "family": "b", "group": "g", "split": "public", "state": {"k": 1}, "labels": ["no", "yes"], "expected": "no",
         "question": {"type": "noul", "instructions": "Allowed?", "criteria": {"true": "All met.", "false": "Something missing."}},
         "provenance": {"license": "MIT"}},
        {"id": "hard-b-1", "family": "b", "group": "g", "split": "public", "state": "T", "labels": ["0", "1"], "expected": 1,
         "question": {"type": "score", "instructions": "Rate.", "criteria": ["low", "high"]}, "provenance": {"license": "MIT"}},
    ]
    (items / "easy.jsonl").write_text(json.dumps(rows[0]) + "\n")
    (items / "hard.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[1:]))
    runs = tmp_path / "runs"
    for model, answers in {"m1": {"easy-a-0": ("y", {"x": 0.2, "y": 0.8}), "hard-b-0": ("yes", {"yes": 0.6, "no": 0.4})},
                           "m2": {"easy-a-0": ("x", {"x": 0.7, "y": 0.3}), "hard-b-0": (None, None)}}.items():
        d = runs / "jevbench-x" / model
        d.mkdir(parents=True)
        d.joinpath("results.jsonl").write_text("".join(json.dumps(
            {"task_id": t, "predicted": p, "probs": pr, "correct": p is not None and p == {"easy-a-0": "y", "hard-b-0": "no"}[t],
             "status": "ok" if p else "failed", "error": None if p else "HTTP 422"}) + "\n" for t, (p, pr) in answers.items()))
    loaded = benchdata.load_items(items)
    assert set(loaded) == {"easy-a-0", "hard-b-0", "hard-b-1"} and loaded["hard-b-0"]["tier"] == "hard"
    assert benchdata.options(loaded["hard-b-0"]) == [["yes", "All met."], ["no", "Something missing."]]
    assert benchdata.options(loaded["hard-b-1"]) == [["0", "low"], ["1", "high"]]

    listing = benchdata.item_list(runs, items)
    by_id = {r["id"]: r for r in listing["items"]}
    assert listing["models"] == 2 and by_id["easy-a-0"]["correct"] == 1 and by_id["easy-a-0"]["p_expected"] == 0.55
    d = benchdata.item_detail("hard-b-0", runs, items)
    assert d["expected"] == "no" and d["siblings"] == ["hard-b-1"]
    assert [m["name"] for m in d["models"]] == ["m1", "m2"] and d["models"][1]["predicted"] is None
    assert d["models"][0]["p_expected"] == 0.4


def test_item_routes():
    site = Site(None)
    code, _, body = site.route("/arenas/bench/api/items")
    assert code == 200 and len(json.loads(body)["items"]) == 231
    first = json.loads(body)["items"][0]["id"]
    assert site.route(f"/arenas/bench/api/item/{first}")[0] == 200
    assert site.route("/arenas/bench/api/item/nope")[0] == 404
    assert site.route("/arenas/bench/items/")[0] == 200 and site.route("/arenas/bench/items")[0] == 301
