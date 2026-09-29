# Brief: deploy small Jev-like models and run the JevBench public items (231)

Project: /home/intuser/li/dev/arena_jev (a UNO arena for "Jev-like" decision models). Machine: NVIDIA GB10
(DGX Spark), aarch64, CUDA 13 (driver 580), 119 GB unified memory shared by everything, 3 TB disk.
Four agents work at the same time, each on its own models and ports. The GPU is shared: keep at most
two of your servers running at once and stop each when its model is done.

## Goal for each of your models
1. Serve it on its port with a TypeSafe-style `POST /v1/systemone` endpoint, using the AUTHORS' OWN inference
   code and prompt format (their pip package / repo at a pinned commit / model-card snippet). Do not invent a
   prompt format or change the model. If the authors ship a /v1/systemone server, use it as is.
   Otherwise write `servers/adapters/<module>.py` for `servers/adapters/serve.py` (read its docstring: define
   NAME, load(device) returning the torch modules, choice(state, instructions, options) -> probabilities, and,
   when the authors support them natively, noul(...) and score(...); plus count_tokens and MAX_INPUT_TOKENS).
   `servers/adapters/example_uniform.py` is a minimal example. choice() must accept ANY number of options
   (the UNO games send 2-15, sometimes more); if the model has a hard option cap, say so (max_options).
2. Describe it in `servers/models/<name>.toml` (see servers/launch.py docstring and servers/models/kev-0.8b.toml):
   name, port, cwd, command, optional [env]; add these top-level keys for the arena:
     tokenizer = "<HF repo whose tokenizer the model uses>"
     max_input_tokens = <int: tokens the model can take for our request text (state + instructions + options),
                         i.e. its real input limit minus the wrapper/prompt overhead its code adds; be conservative>
     max_options = <int or omit if unlimited>
     notes = "<one line: code path, revision/commit, anything unusual>"
3. Start / check / stop only through the launcher (it adds the parameter probe and pid files):
     cd /home/intuser/li/dev/arena_jev
     python3 servers/launch.py start <name> ; python3 servers/launch.py wait <name> ; python3 servers/launch.py stop <name>
   NEVER use `pkill -f` / `pgrep -f` with a pattern that also appears in your own command line (it kills your
   shell). Stop servers with launch.py stop.
4. Quick check, then the full run (231 items, official harness, one request at a time):
     cd third_party/jevbench && ../../.venv/bin/python -m jevbench.cli run --tasks ../../runs/jevbench-public-20260929/public_all.jsonl \
        --adapter typesafe --endpoint http://127.0.0.1:<port> --key-env '' --model <name> --reserve-usd 0 --limit 5 \
        --results /tmp/claude-1000/<something>/r.jsonl --raw-dir /tmp/claude-1000/<something>/raw --ledger /tmp/claude-1000/<something>/l.jsonl
     cd /home/intuser/li/dev/arena_jev && bench/run_jevbench.sh runs/jevbench-public-20260929 <name>=<port>
   (the script refuses to reuse a result directory; delete runs/jevbench-public-20260929/<name>/ only if YOUR
   earlier attempt for that model failed and must be redone.) Check runs/jevbench-public-20260929/<name>.run.log:
   "231/231 attempted, 0 failed". A few failures on over-long items are acceptable if the model truly cannot take
   them (the server must answer HTTP 422 "maximum context length", never truncate silently); report them.
5. Parameter count, "measured": servers/logs/<name>.params.json (written by the probe for launched servers, or by
   serve.py for adapters). Definition: parameters held after loading with the authors' code, every tensor once,
   LoRA counted as merged if the code merges it; speed-ups that turn weights into non-parameters (kernel fusion)
   must be off for the measurement. If the probe's number looks wrong, find out why and fix the measurement.
6. Stop the server when done (the UNO tournament later restarts all of them together from the .toml files).

## Environment rules
- Python environments: create your own venvs under servers/venvs/ named `<agentletter>-<something>` (e.g.
  `a-qwen`, `c-laya`) or `<model>`; you may share one of YOUR venvs across your models when compatible. Never pip
  install into a venv you did not create (servers/venvs/{kev,decider,jevstyle,openthai} and .venv are off-limits).
  torch: `pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130` (cached locally, works on
  the GB10, sm_121). Qwen3.5 models want `flash-linear-attention==0.5.2`; causal-conv1d is not needed.
- Code: clone into servers/src/<repo> (git clone; pin and note the commit). Avoid the GitHub REST API (60 req/h
  unauthenticated, shared); use git clone or raw.githubusercontent.com.
- Hugging Face: a read token is saved in ~/.cache/huggingface/token (used automatically). Gated google/gemma-3-270m
  is approved for it.
- Do not edit: arena/, bench/models.json, bench/run_jevbench.sh, servers/launch.py, servers/adapters/serve.py,
  other agents' files, anything outside the project (except /tmp/claude-1000 scratch). No sudo.
- Reference: third_party/jevbench/jevbench/adapters/ has the JevBench maintainers' adapters for some families
  (gliner2_local, laya_local, verdict_local, local_openjev, semif_direct, so1_decider, ...), useful to see how
  those models were run officially. docs/jev_models.md has notes on most models.
- Time box: about 25 minutes of effort per model. If a model cannot be run faithfully (install impossible on
  aarch64/CUDA 13, missing weights, code broken), stop, clean up its running server, and report why.

## Final answer: a JSON array, one object per model:
{"name", "status": "ok" | "failed", "reason" (if failed), "port", "toml": "servers/models/<name>.toml",
 "code_path": "package/repo@commit + entry point", "params": int, "params_source": "...",
 "tokenizer", "max_input_tokens", "max_options", "jevbench": {"acc", "hard", "attempted", "failed", "p50_ms"},
 "notes": "anything the arena should know (e.g. only fp32 works, needs N GB, slow, noul/score via choice)"}
