# arena_jev

Jev-like decision models (TypeSafe Jev, and open models that serve `POST /v1/systemone`) play games against each
other, so their abilities can be compared by results. The current game is 4-player UNO.

A survey of the models is in [`docs/jev_models.md`](docs/jev_models.md); JevBench public-item results for 28 small
open models are in [`runs/jevbench-public-20260929/`](runs/jevbench-public-20260929/README.md), and the items
themselves in [`data/jevbench-public/`](data/jevbench-public/README.md).

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'   # the platform needs trueskill and tokenizers; tests use pytest
.venv/bin/python -m pytest                                   # rules engine, prompt building, client, tournament

# sanity check without models: 1 heuristic player against 3 random players, 1000 games in a few seconds
.venv/bin/python -m arena run configs/sanity.toml

# 4 local 0.8B models at one table
servers/setup.sh          # one venv per model (their dependencies conflict); run once
servers/start.sh          # start the servers: kev 8101 · decider 8102 · jevstyle 8103 · openthai 8104
.venv/bin/python servers/check.py                        # send one test request to each server
.venv/bin/python -m arena run configs/local4.toml        # 250 deals x 4 seat rotations = 1000 games
servers/stop.sh

# results and replays
.venv/bin/python -m arena viz runs/<dir>                         # http://127.0.0.1:8200
.venv/bin/python -m arena viz runs/<dir> --export view.html      # one self-contained HTML file (replays of the first 3 deals)
.venv/bin/python -m arena prompt --seed 3 --step 30              # the request a model receives at some point of a game
```

## Swiss tournament: all 28 models (locally or on a cluster)

The server environments (venvs, third-party sources, weight revisions) are described in
[`servers/envs/README.md`](servers/envs/README.md); each model's start command is in `servers/models/<name>.toml`.

```bash
export HF_HUB_CACHE=~/.cache/huggingface/hub         # weight cache (the model files refer to this variable)
export ARENA_GPUS=0,1,2,3,4,5,6,7                    # several GPUs: models are spread over them by name (not needed for one GPU)
python3 servers/launch.py start && python3 servers/launch.py wait
.venv/bin/python -m arena swiss configs/swiss.toml > runs/swiss.log 2>&1 &   # results go to runs/<time>_swiss/
```

- Rules and settings are in `configs/swiss.toml`: 4-player games, Wild Draw Four challenge, hand listed in a shuffled
  order, must-play (a house rule: without it, weak models that keep drawing stall games forever), 1000 games per
  model, a maximum lead of 40 games, at most 1000 decisions per game.
- Continue a stopped run: `.venv/bin/python -m arena swiss configs/swiss.toml --resume runs/<run>`.
- Restart one server during a run: add its name to `runs/<run>/hold.txt`, wait until `status.json` no longer lists it
  under `busy`, run `servers/launch.py stop/start/wait`, then remove it from `hold.txt`.
- Publish results: `python3 bench/pack_runs.py pack runs/<run>` (the jsonl files become xz, split above 45 MB; the raw
  jsonl files are not committed), then commit `runs/<run>/`.
- On the machine that shows the results: `git pull`, `python3 bench/pack_runs.py unpack runs/<run>`, then start the
  site with `.venv/bin/python -m arena site --uno-run latest --port 8200` (the UNO page uses the newest `*_swiss` run).

## Connect Four: two models against each other

```bash
python3 servers/launch.py start kev-0.8b decider-0.8b && python3 servers/launch.py wait kev-0.8b decider-0.8b
.venv/bin/python -m arena c4 kev-0.8b decider-0.8b              # 7 openings x both colors = 14 games
.venv/bin/python -m arena c4 kev-0.8b heuristic --openings 49   # 98 games; players: model names, random, heuristic
.venv/bin/python -m arena c4-show runs/<dir> --game G03-ab      # replay a game in the terminal
```

- Standard rules (7 columns x 6 rows, four in a row wins, a full board is a draw), engine in `arena/connect4/`.
- The board is drawn from the mover's point of view (X is always "you"); options are the legal columns with keys
  "1".."7" and say where the disc lands, with no tactical hints. The move list is dropped only when a request would
  exceed the model's own input limit.
- Each opening (none, every first move, or every first two moves) is played once with each player moving first.
- Per player: wins, draws and losses (overall, as first and as second), and three checks that need no solver: took an
  immediate win when one existed, blocked the opponent's immediate win, let the opponent win on top of its own disc.
- Output: `runs/<time>_c4-<a>-vs-<b>/` with `games.jsonl`, `decisions.jsonl` (options, probabilities, latency per
  move) and `summary.md`.

## Layout

| File | Purpose |
|---|---|
| `arena/uno/cards.py` | the classic 108-card deck |
| `arena/uno/engine.py` | rules engine (official rules; optional Wild Draw Four challenge and must-play house rule); accepts legal actions only |
| `arena/uno/render.py` | renders the position as a plain-text state and builds the questions; rules templates A/B/C, `history_rounds` rounds of history |
| `arena/players.py` | random, heuristic and System One players; asks for the card first and for the color after a wild, one request per question |
| `arena/systemone.py` | `/v1/systemone` client |
| `arena/match.py`, `tournament.py` | one game, fixed schedules with seat rotations, parallel execution, logging |
| `arena/swiss.py` | Swiss tournament over all model servers, rated with TrueSkill (resume, hold, maximum lead) |
| `arena/stats.py`, `leaderboard.py` | win rates with Wilson 95% intervals, points, diagnostics, the leaderboard |
| `arena/connect4/` | Connect Four: engine, prompts, players, two-player matches, terminal replay |
| `arena/replay.py`, `viz.py`, `viewer.html` | replays: re-simulate a game from its seed and logged actions, and rebuild the requests sent to the models |
| `arena/site.py`, `arena/static/` | the site: home, leaderboard, JevBench results and the UNO viewer under `/arenas/` |
| `servers/` | model servers: launcher, adapters, one file per model, environment manifests |
| `bench/` | model registry with measured sizes, JevBench runner, deployment reports, run packing |

## Fairness

- Every player receives the same request text: the state is plain text rendered by the platform (servers serialize
  JSON objects differently).
- The platform shuffles the options with a seed, so every order can be reproduced. A decision with a single legal
  option is taken without calling the model.
- Each deal is played once per seat rotation, which cancels out the luck of the deal and the seat. With 4 identical
  heuristic players, P0 wins 26.6% and P3 23.4%.
- A model's choice is the option with the highest value in `probabilities`, not `confidence` (servers define
  confidence differently).
- Output directory `runs/<time>_<name>/`:
  - `games.jsonl`: one line per game, with all events and the starting hands;
  - `decisions.jsonl`: one line per decision, with the option probabilities, latency, token counts and diagnostic flags;
  - `summary.md`: the summary.
