#!/usr/bin/env bash
# Build one virtualenv per local model server. Their pinned dependencies conflict
# (kev wants numpy>=2.5 and torch<2.9, decider wants numpy<2), so they cannot share one.
# Every venv gets torch 2.13.0+cu130, the build that runs on the GB10 (sm_121).
#
#   servers/setup.sh [kev|decider|jevstyle|openthai ...]   (default: all four)
set -euo pipefail
cd "$(dirname "$0")"
TORCH="torch==2.13.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

mkvenv() {
  [ -x "venvs/$1/bin/python" ] || python3 -m venv "venvs/$1"
  "venvs/$1/bin/pip" install -q --upgrade pip
  "venvs/$1/bin/pip" install -q "$TORCH" --index-url "$TORCH_INDEX"
}

clone() {  # url dir [ref]
  [ -d "src/$2/.git" ] || git clone -q "$1" "src/$2"
  if [ -n "${3:-}" ]; then git -C "src/$2" checkout -q "$3"; fi
  echo "$2 @ $(git -C "src/$2" rev-parse --short HEAD)"
}

setup_kev() {
  mkvenv kev
  clone https://github.com/jaredpalmer/kev kev 3e1cd3bb588a388a06827443380befece23e68c7
  # kev pins torch<2.9, which has no GB10 build: install it without deps and add the runtime deps by hand.
  venvs/kev/bin/pip install -q --no-deps -e src/kev
  venvs/kev/bin/pip install -q "transformers>=5.17,<6" "peft>=0.21" "accelerate>=1.15" "pydantic>=2.9" \
    numpy fastapi uvicorn flash-linear-attention huggingface_hub
}

setup_decider() {
  mkvenv decider
  clone https://github.com/Mapika/decider decider
  venvs/decider/bin/pip install -q -e "src/decider[serve]"
}

setup_jevstyle() {
  mkvenv jevstyle
  venvs/jevstyle/bin/pip install -q "jev-style[torch]==0.3.0"
}

setup_openthai() {
  mkvenv openthai
  clone https://github.com/iapp-technology/openthai-systemone openthai
  venvs/openthai/bin/pip install -q -e "src/openthai[server]" flash-linear-attention
}

for name in "${@:-kev decider jevstyle openthai}"; do
  for n in $name; do
    echo "=== $n"
    "setup_$n"
    echo "=== $n done"
  done
done
