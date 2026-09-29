#!/usr/bin/env bash
# Stop servers started by start.sh:  servers/stop.sh [name ...]
cd "$(dirname "$0")"
for n in ${@:-kev decider jevstyle openthai}; do
  if [ -f "logs/$n.pid" ]; then kill "$(cat "logs/$n.pid")" 2>/dev/null && echo "stopped $n"; rm -f "logs/$n.pid"; fi
done
