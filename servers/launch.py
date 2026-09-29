"""Start, stop and check local model servers described by servers/models/<name>.toml.

    python servers/launch.py start  [name ...]    (default: all)
    python servers/launch.py stop   [name ...]
    python servers/launch.py status [name ...]
    python servers/launch.py wait   [name ...]    block until each answers a small request (or fails)

A model file:

    name = "kev-0.8b"                          # registry name (bench/models.json)
    port = 8101
    cwd = "src/kev"                            # relative to servers/ (optional)
    command = ["venvs/kev/bin/python", "-m", "kev.serve", "--run", "jaredpalmer/kev-0.8b", "--port", "8101"]
    [env]                                      # optional extra environment
    KEV_FUSED = "1"

Relative paths in `command` that exist under servers/ are made absolute. `~` and environment variables such as
${HF_HUB_CACHE} (default ~/.cache/huggingface/hub) are expanded in `command` and `cwd`. Every server gets the
parameter probe (bench/paramprobe, writes logs/<name>.params.json); logs go to logs/<name>.log, the pid to
logs/<name>.pid.

Several GPUs: set ARENA_GPUS="0,1,2,3" and the servers are spread over them round-robin (CUDA_VISIBLE_DEVICES, in
the sorted order of all model files, so a model always lands on the same GPU); `gpu = 2` in a model file pins it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
PROBE = HERE.parent / "bench" / "paramprobe"
PING = {
    "model": "local",
    "state": "You are P1 in a 4-player game of UNO.\nYour hand (2 cards): Red 5, Wild\nTop card: Green 5 (current color: Green)",
    "questions": {"move": {"type": "choice", "instructions": "Choose the move that gives you the best chance of winning.",
                           "criteria": {"red_5": "Play Red 5.", "wild": "Play Wild.", "draw": "Draw 1 card."}}},
}


def models(names: list[str]) -> list[dict]:
    specs = {}
    for f in sorted((HERE / "models").glob("*.toml")):
        with open(f, "rb") as fh:
            spec = tomllib.load(fh)
        specs[spec["name"]] = spec
    if not names:
        return list(specs.values())
    unknown = [n for n in names if n not in specs]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}; known: {sorted(specs)}")
    return [specs[n] for n in names]


def pid_of(name: str) -> int | None:
    f = LOGS / f"{name}.pid"
    if not f.exists():
        return None
    pid = int(f.read_text())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        return None


def gpu_for(name: str, spec: dict) -> str | None:
    if "gpu" in spec:
        return str(spec["gpu"])
    gpus = [g.strip() for g in os.environ.get("ARENA_GPUS", "").split(",") if g.strip()]
    if not gpus:
        return None
    names = sorted(f.stem for f in (HERE / "models").glob("*.toml"))
    return gpus[names.index(name) % len(gpus)] if name in names else gpus[0]


def start(spec: dict) -> None:
    name = spec["name"]
    if pid_of(name):
        print(f"{name}: already running (pid {pid_of(name)})")
        return
    LOGS.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    expand = lambda c: os.path.expanduser(os.path.expandvars(c))  # noqa: E731
    cmd = [str(HERE / c) if (HERE / c).exists() and not c.startswith("-") else c for c in map(expand, spec["command"])]
    # a config file from models/ (e.g. a yaml the server reads itself) is passed as a copy with ${VARS} expanded
    for i, c in enumerate(cmd):
        f = Path(c)
        if f.suffix in (".yaml", ".yml", ".json") and f.is_file() and f.parent == HERE / "models":
            copy = LOGS / f"{name}.{f.name}"
            copy.write_text(os.path.expandvars(f.read_text()))
            cmd[i] = str(copy)
    env = {**os.environ, **{k: str(v) for k, v in spec.get("env", {}).items()}}
    gpu = gpu_for(name, spec)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = f"{PROBE}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(PROBE)
    env["ARENA_PARAM_PROBE"] = str(LOGS / f"{name}.params.json")
    cwd = HERE / expand(spec["cwd"]) if spec.get("cwd") else HERE
    with open(LOGS / f"{name}.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (LOGS / f"{name}.pid").write_text(str(proc.pid))
    print(f"{name}: pid {proc.pid} port {spec['port']}")


def stop(spec: dict) -> None:
    pid = pid_of(spec["name"])
    if pid:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"{spec['name']}: stopped")
    (LOGS / f"{spec['name']}.pid").unlink(missing_ok=True)


def ping(spec: dict, timeout: float = 600) -> tuple[bool, str]:
    req = urllib.request.Request(f"http://127.0.0.1:{spec['port']}/v1/systemone", data=json.dumps(PING).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ans = json.load(r)["answers"]["move"]
        return True, json.dumps(ans.get("probabilities"))
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def wait(spec: dict, limit: float = 900) -> bool:
    start_t = time.time()
    while time.time() - start_t < limit:
        if not pid_of(spec["name"]):
            print(f"{spec['name']}: process exited; see {LOGS / (spec['name'] + '.log')}")
            return False
        ok, msg = ping(spec, timeout=30)
        if ok:
            print(f"{spec['name']}: ready in {time.time() - start_t:.0f}s {msg}")
            return True
        time.sleep(5)
    print(f"{spec['name']}: not ready after {limit:.0f}s")
    return False


def main() -> None:
    cmd, names = sys.argv[1], sys.argv[2:]
    for spec in models(names):
        if cmd == "start":
            start(spec)
        elif cmd == "stop":
            stop(spec)
        elif cmd == "wait":
            wait(spec)
        elif cmd == "status":
            pid = pid_of(spec["name"])
            ok, msg = ping(spec, timeout=10) if pid else (False, "not running")
            print(f"{spec['name']:28s} port {spec['port']}  pid {pid or '-':>8}  {'OK ' if ok else '-- '}{msg[:80]}")
        else:
            sys.exit("usage: launch.py start|stop|status|wait [name ...]")


if __name__ == "__main__":
    main()
