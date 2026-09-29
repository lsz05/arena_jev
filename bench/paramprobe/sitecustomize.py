"""Parameter probe for model servers: counts the parameters a serving process actually holds.

Enabled by ARENA_PARAM_PROBE=<output json>. A daemon thread polls every 5 s: it walks every torch.nn.Module
alive in the process, collects their own parameters (shared/tied tensors counted once) and writes the total,
plus the top-level module classes, to the output file whenever it changes. It gives up after 15 minutes
without change. Put this directory on PYTHONPATH so Python imports it at startup.
"""

import os

_OUT = os.environ.get("ARENA_PARAM_PROBE")

if _OUT:
    import gc
    import json
    import sys
    import threading
    import time

    def _count():
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        modules = [o for o in gc.get_objects() if isinstance(o, torch.nn.Module)]
        params, children = {}, set()
        for m in modules:
            for p in m.parameters(recurse=False):
                params[id(p)] = p.numel()
            children.update(id(c) for c in m._modules.values() if c is not None)
        top = {}
        for m in modules:
            if id(m) not in children:
                n = sum(p.numel() for p in m.parameters())
                if n:
                    top[type(m).__name__] = top.get(type(m).__name__, 0) + n
        return {"params": sum(params.values()), "tensors": len(params), "top_level": top}

    def _probe():
        last, last_change = None, time.time()
        while time.time() - last_change < 900:
            time.sleep(5)
            try:
                result = _count()
            except Exception as e:  # never disturb the server
                result = {"error": repr(e)}
            if result and result != last and result.get("params", 1):
                last, last_change = result, time.time()
                with open(_OUT + ".tmp", "w") as f:
                    json.dump({**result, "pid": os.getpid(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                os.replace(_OUT + ".tmp", _OUT)

    threading.Thread(target=_probe, name="arena-param-probe", daemon=True).start()
