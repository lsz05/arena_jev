import json

from arena.match import play_game
from arena.mock_server import serve
from arena.players import HeuristicPlayer, RandomPlayer, SystemOnePlayer
from arena.replay import frames
from arena.systemone import SystemOneClient


def test_replay_reproduces_game_and_prompts():
    server = serve(port=0, background=True)
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        model = SystemOnePlayer("mock", SystemOneClient(url), history_rounds=3)
        players = [model, HeuristicPlayer("h"), RandomPlayer("r"), HeuristicPlayer("h2")]
        for seed in range(5):
            decisions = []
            record = play_game(players, seed, game_id=f"g{seed}", on_decision=decisions.append)
            record = json.loads(json.dumps(record))  # as read back from games.jsonl
            decisions = json.loads(json.dumps(decisions))
            specs = {"mock": {"name": "mock", "type": "systemone", "history_rounds": 3}, "h": {"type": "heuristic"}}
            replay = frames(record, decisions, specs, {"template": "B"})
            assert len(replay["frames"]) == len(decisions) + 1
            assert replay["frames"][-1]["kind"] == "end"
            prompts = [f["prompt"] for f in replay["frames"] if "prompt" in f]
            assert prompts and all(p["matches_log"] for p in prompts)
            end = replay["frames"][-1]
            assert end["hand_points"] == record["hand_points"]
    finally:
        server.shutdown()
