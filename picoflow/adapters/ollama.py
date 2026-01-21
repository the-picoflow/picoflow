from __future__ import annotations

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


def ollama_factory(u: urllib.parse.ParseResult, qs: Dict[str, str]) -> LLMAdapter:
    """
    DSN:
      llm+ollama://host:port/model?k=v

    Examples:
      llm+ollama://localhost:11434/llama3.1
      llm+ollama://127.0.0.1:11434/qwen2.5?temperature=0.2&timeout=60
    """
    model = (u.path or "").lstrip("/")
    if not model:
        raise ValueError("Model missing in DSN path, e.g. llm+ollama://localhost:11434/MODEL")

    host = u.netloc.strip() or "localhost:11434"

    # Ollama default API base path is /api
    base_path = (qs.get("base_path") or "/api").rstrip("/")
    endpoint = f"http://{host}{base_path}/chat"

    timeout = _maybe_float(qs.get("timeout"))  # seconds

    # Common sampling options (Ollama uses `options`)
    temperature = _maybe_float(qs.get("temperature"))
    top_p = _maybe_float(qs.get("top_p"))
    num_predict = _maybe_int(qs.get("max_tokens") or qs.get("num_predict"))
    keep_alive = qs.get("keep_alive")  # e.g. "5m" or "0" (optional)

    def _headers() -> Dict[str, str]:
        return {"Content-Type": "application/json"}

    def _body(prompt: str, stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }

        opts: Dict[str, Any] = {}
        if temperature is not None:
            opts["temperature"] = temperature
        if top_p is not None:
            opts["top_p"] = top_p
        if num_predict is not None:
            opts["num_predict"] = num_predict

        if opts:
            payload["options"] = opts
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        return payload

    async def _post_json(payload: Dict[str, Any]) -> Dict[str, Any]:
        def _do() -> Dict[str, Any]:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=_headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                raise_http_error(
                    e,
                    provider="ollama",
                    hint="Is Ollama running? Check host/port and model name."
                )
            except urllib.error.URLError as e:
                raise_url_error(
                    e,
                    provider="ollama",
                    hint="Is Ollama running? Check host/port and model name."
                )

        return await run_blocking(_do)

    async def _post_stream(payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Ollama streaming returns NDJSON: one JSON object per line.
        Each chunk often contains: {"message":{"content":"..."}, ...}
        Final chunk includes: {"done":true, ...}
        """

        def _open():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=_headers(), method="POST")
            return urllib.request.urlopen(req, timeout=timeout)

        # ---- open connection ----
        try:
            resp = await run_blocking(_open)
        except urllib.error.HTTPError as e:
            raise_http_error(
                e,
                provider="ollama",
                hint="Is Ollama running? Check host/port and model name."
            )
        except urllib.error.URLError as e:
            raise_url_error(
                e,
                provider="ollama",
                hint="Is Ollama running? Check host/port and model name."
            )

        # ---- read stream ----
        try:
            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, resp.readline)
                if not line:
                    break

                s = line.decode("utf-8", errors="ignore").strip()
                if not s:
                    continue

                try:
                    obj = json.loads(s)
                except Exception:
                    continue

                msg = obj.get("message") or {}
                piece = msg.get("content")
                if isinstance(piece, str) and piece:
                    yield piece

                if obj.get("done") is True:
                    break

        except urllib.error.HTTPError as e:
            raise_http_error(
                e,
                provider="ollama",
                hint="Stream interrupted; check Ollama status, host/port, and model name."
            )
        except urllib.error.URLError as e:
            raise_url_error(
                e,
                provider="ollama",
                hint="Stream interrupted; check Ollama status and network."
            )
        except Exception as e:
            raise RuntimeError(f"[ollama] Stream interrupted: {e}") from None

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
            # Non-stream response typically: {"message":{"content":"..."}}
            msg = obj.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str):
                return content
            # Fallback: show raw JSON if schema differs
            return json.dumps(obj, ensure_ascii=False)

        return _one()

    return adapter


register_llm_provider("ollama", ollama_factory)
