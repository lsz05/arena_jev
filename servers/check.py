"""Send one small UNO-shaped request to each server and print the answer: python servers/check.py [port ...]"""
import json
import sys
import time
import urllib.request

PORTS = {8101: "kev", 8102: "decider", 8103: "jevstyle", 8104: "openthai"}
REQUEST = {
    "model": "jev-latest",
    "state": "You are P1 in a 4-player game of UNO.\nYour hand (3 cards): Red 5, Blue 2, Wild\nTop card: Green 5 (current color: Green)",
    "questions": {"move": {"type": "choice", "instructions": "Choose the move that gives you the best chance of winning.",
                           "criteria": {"red_5": "Play Red 5 (matches the 5).", "wild": "Play Wild and name the next color.",
                                        "draw": "Draw 1 card instead of playing."}}},
}
for port in [int(p) for p in sys.argv[1:]] or PORTS:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(REQUEST).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        print(f"{PORTS.get(port, port)}: {time.time() - t:.2f}s {json.dumps(out['answers']['move'])}")
    except Exception as e:
        print(f"{PORTS.get(port, port)}: ERROR {e}")
