from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
import urllib.error
import asyncio
from typing import Any, Dict, Optional, AsyncGenerator

from .types import LLMAdapter
from .registry import register_llm_provider
from ._http import run_blocking, raise_http_error, raise_url_error


def _maybe_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"Invalid float: {v}")


def _maybe_int(v: Optional[str]) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"Invalid int: {v}")


def openai_compat_factory(u: urllib.parse.ParseResult, qs: Dict[str, str]) -> LLMAdapter:
    model = (u.path or "").lstrip("/")
    if not model:
        raise ValueError("Model missing in DSN path, e.g. llm+openai://host/MODEL?....")

    host = u.netloc.strip()
    base_url = qs.get("base_url")
    if base_url:
        base = base_url.rstrip("/")
    else:
        if host:
            # default scheme http for local, https for api.openai.com
            if host == "api.openai.com":
                base = "https://api.openai.com"
            else:
                base = f"http://{host}"
        else:
            base = "https://api.openai.com"

    base_path = qs.get("base_path", "/v1").rstrip("/")
    endpoint = f"{base}{base_path}/chat/completions"

    api_key = qs.get("api_key")
    if not api_key:
        env = qs.get("api_key_env")
        if env:
            api_key = os.environ.get(env, "")
    if api_key == "none":
        api_key = ""

    temperature = _maybe_float(qs.get("temperature"))
    top_p = _maybe_float(qs.get("top_p"))
    max_tokens = _maybe_int(qs.get("max_tokens"))
    timeout = _maybe_float(qs.get("timeout"))

    def _headers() -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _body(prompt: str, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def _post_json(payload):
        def _do():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=_headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                raise_http_error(
                    e,
                    provider="openai",
                    hint="Check API key, model name, and endpoint."
                )
            except urllib.error.URLError as e:
                raise_url_error(
                    e,
                    provider="openai",
                    hint="Check network and base_url/host."
                )

        return await run_blocking(_do)

    async def _post_stream(payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        def _open():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=_headers(), method="POST")
            return urllib.request.urlopen(req, timeout=timeout)

        # ---- open connection (already handled) ----
        try:
            resp = await run_blocking(_open)
        except urllib.error.HTTPError as e:
            raise_http_error(
                e,
                provider="openai",
                hint="Check API key, model name, and endpoint."
            )
        except urllib.error.URLError as e:
            raise_url_error(
                e,
                provider="openai",
                hint="Check network and base_url/host."
            )

        # ---- read stream (NEW: add except) ----
        try:
            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, resp.readline)
                if not line:
                    break
                s = line.decode("utf-8", errors="ignore").strip()
                if not s or not s.startswith("data:"):
                    continue
                data = s[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj["choices"][0].get("delta", {})
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        yield piece
                except Exception:
                    continue

        except urllib.error.HTTPError as e:
            raise_http_error(
                e,
                provider="openai",
                hint="Stream interrupted; check API key/model/endpoint."
            )
        except urllib.error.URLError as e:
            raise_url_error(
                e,
                provider="openai",
                hint="Stream interrupted; check network and base_url/host."
            )
        except Exception as e:
            raise RuntimeError(f"[openai] Stream interrupted: {e}") from None

        finally:
            try:
                resp.close()
            except Exception:
                pass

    def adapter(prompt: str, stream: bool):
        if stream:
            return _post_stream(_body(prompt, True))

        async def _one() -> str:
            obj = await _post_json(_body(prompt, False))
            try:
                return obj["choices"][0]["message"]["content"]
            except Exception:
                return json.dumps(obj, ensure_ascii=False)

        return _one()

    return adapter


# register aliases
register_llm_provider("openai", openai_compat_factory)
register_llm_provider("openai_compat", openai_compat_factory)
