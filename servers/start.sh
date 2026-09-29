#!/usr/bin/env bash
# Start local model servers on 127.0.0.1:
#   kev 8101 · decider 8102 · jevstyle 8103 · openthai 8104
#   servers/start.sh [kev|decider|jevstyle|openthai ...]   (default: all four)
# Logs: servers/logs/<name>.log   PIDs: servers/logs/<name>.pid   Stop: servers/stop.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
HERE=$(pwd)
# parameter probe: every server writes logs/<name>.params.json with the parameters it actually loaded
export PYTHONPATH="$HERE/../bench/paramprobe${PYTHONPATH:+:$PYTHONPATH}"

launch() {  # name port command...
  local name=$1 port=$2; shift 2
  if [ -f "logs/$name.pid" ] && kill -0 "$(cat "logs/$name.pid")" 2>/dev/null; then
    echo "$name already running (pid $(cat "logs/$name.pid"))"; return
  fi
  ARENA_PARAM_PROBE="$HERE/logs/$name.params.json" nohup "$@" > "logs/$name.log" 2>&1 &
  echo $! > "logs/$name.pid"
  echo "$name: pid $! port $port"
}

start_kev() {
  (cd src/kev && launch_in_dir kev 8101 "$HERE/venvs/kev/bin/python" -m kev.serve --run jaredpalmer/kev-0.8b --port 8101)
}
start_decider() {
  (cd src/decider && DECIDER_MODEL=Mapika/decider-0.8b launch_in_dir decider 8102 \
    "$HERE/venvs/decider/bin/uvicorn" decider.serve:app --host 127.0.0.1 --port 8102)
}
start_jevstyle() {
  launch jevstyle 8103 venvs/jevstyle/bin/jev-style serve --port 8103 --backend torch --device cuda
}
start_openthai() {
  OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne launch openthai 8104 \
    venvs/openthai/bin/uvicorn openthai_systemone.server:app --host 127.0.0.1 --port 8104
}

launch_in_dir() {  # like launch, but logs/pids live in servers/logs while cwd is the source dir
  local name=$1 port=$2; shift 2
  if [ -f "$HERE/logs/$name.pid" ] && kill -0 "$(cat "$HERE/logs/$name.pid")" 2>/dev/null; then
    echo "$name already running"; return
  fi
  ARENA_PARAM_PROBE="$HERE/logs/$name.params.json" nohup "$@" > "$HERE/logs/$name.log" 2>&1 &
  echo $! > "$HERE/logs/$name.pid"
  echo "$name: pid $! port $port"
}

for n in ${@:-kev decider jevstyle openthai}; do "start_$n"; done
