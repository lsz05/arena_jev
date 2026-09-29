#!/usr/bin/env bash
# Run the JevBench public items (easy 48 + original 72 + hard 111 = 231) against the local
# model servers with the official harness (third_party/jevbench, typesafe adapter, one request at a
# time, no retries), then summarize each model with the official scorer.
#   bench/run_jevbench.sh OUT_DIR [name=port ...]    (servers must be running: servers/start.sh)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(realpath -m "$1"); shift
PY="$ROOT/.venv/bin/python"
JB="$ROOT/third_party/jevbench"
MODELS=("${@:-kev-0.8b=8101 decider-0.8b=8102 jev-style-0.8b=8103 openthai-0.8b=8104}")
mkdir -p "$OUT"
[ -f "$OUT/public_all.jsonl" ] || cat "$JB"/datasets/public/{easy,original,hard}.jsonl > "$OUT/public_all.jsonl"
cd "$JB"
for spec in ${MODELS[@]}; do
  name=${spec%%=*}; port=${spec#*=}
  ( "$PY" -m jevbench.cli run --tasks "$OUT/public_all.jsonl" --adapter typesafe \
      --endpoint "http://127.0.0.1:$port" --key-env '' --model "$name" \
      --cost-basis local_gpu_gb10 --reserve-usd 0 --run-label "$name" \
      --results "$OUT/$name/results.jsonl" --raw-dir "$OUT/$name/raw" \
      --ledger "$OUT/$name/ledger.jsonl" --manifest "$OUT/$name/manifest.json" > "$OUT/$name.run.log" 2>&1
    "$PY" -m jevbench.cli summarize --tasks "$OUT/public_all.jsonl" --results "$OUT/$name/results.jsonl" \
      --public-export "$OUT/$name/summary.json" > "$OUT/$name.summary.txt" 2>&1
    echo "$name done" ) &
done
wait
