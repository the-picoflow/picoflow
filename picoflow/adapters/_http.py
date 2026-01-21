# adapters/_http.py
import json
import urllib.error
import asyncio
from typing import Callable, TypeVar

T = TypeVar("T")


async def run_blocking(fn: Callable[[], T]) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


def raise_http_error(e: urllib.error.HTTPError, *, provider: str, hint: str = "") -> None:
    body = None
    try:
        raw = e.read().decode("utf-8", errors="ignore")
        if raw:
            try:
                obj = json.loads(raw)
                # OpenAI-style: {"error": {"message": "..."}}
                if isinstance(obj, dict):
                    if "error" in obj and isinstance(obj["error"], dict):
                        body = obj["error"].get("message")
                    else:
                        body = raw
            except Exception:
                body = raw
    except Exception:
        pass

    msg = f"[{provider}] HTTP {e.code} {e.reason}"
    if body:
        msg += f": {body}"
    if hint:
        msg += f"\nHint: {hint}"

    raise RuntimeError(msg) from None


def raise_url_error(e: urllib.error.URLError, *, provider: str, hint: str = "") -> None:
    msg = f"[{provider}] Network error: {e.reason}"
    if hint:
        msg += f"\nHint: {hint}"
    raise RuntimeError(msg) from None
