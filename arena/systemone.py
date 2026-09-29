"""Minimal client for TypeSafe-style `POST /v1/systemone` servers (Jev and open look-alikes)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class SystemOneError(Exception):
    pass


class SystemOneClient:
    def __init__(self, base_url: str, *, model: str = "jev-latest", api_key: str | None = None,
                 timeout: float = 60.0, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries

    def evaluate(self, state: str | dict | list, questions: dict[str, dict]) -> dict:
        """Send one request; returns the parsed response ({"model", "answers", "usage", ...})."""
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(f"{self.base_url}/v1/systemone", data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                detail = e.read()[:500].decode(errors="replace")
                if e.code < 500 and e.code != 429:  # the request itself is wrong; retrying will not help
                    raise SystemOneError(f"HTTP {e.code}: {detail}") from e
                last = SystemOneError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = e
            if attempt < self.retries:
                time.sleep(0.5 * 2**attempt)
        raise SystemOneError(f"{self.base_url}: failed after {self.retries + 1} attempts: {last}")

    def models(self) -> dict | None:
        """GET /v1/models, for recording what a server reports about itself. None if unsupported."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return None
