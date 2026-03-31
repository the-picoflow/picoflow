from __future__ import annotations

import os
import json
import urllib.request
import urllib.parse
import urllib.error
import asyncio
from typing import Any, Dict, Optional, AsyncGenerator, List

from .types import LLMAdapter
from .registry import register_llm_provider
from ._http import run_blocking, raise_http_error, raise_url_error, raise_timeout_error, is_timeout_error
from ._http import TLSConfig, urlopen_with_tls


def _maybe_float(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"Invalid float: {v}")


def _parse_payload_value(v: str) -> Any:
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return v


def _normalize_messages(prompt: str, messages: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if messages is None:
        return [{"role": "user", "content": prompt}]

    if not isinstance(messages, list):
        raise ValueError("messages must be a list of message objects")

    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{i}] must be an object")
        role = item.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{i}].role must be a non-empty string")
        if "content" not in item:
            raise ValueError(f"messages[{i}].content is required")
        normalized.append(dict(item))
    return normalized


def _response_to_result(obj: Dict[str, Any]) -> LLMResult:
    from ..core import LLMResult

    try:
        message = obj["choices"][0]["message"]
    except Exception:
        return LLMResult(output=json.dumps(obj, ensure_ascii=False))

    if not isinstance(message, dict):
        return LLMResult(output=json.dumps(obj, ensure_ascii=False))

    content = message.get("content")
    if isinstance(content, str):
        return LLMResult(output=content, assistant_message=message)

    return LLMResult(output=json.dumps(obj, ensure_ascii=False), assistant_message=message)


def _is_local_host(host: str) -> bool:
    return host in ("localhost", "127.0.0.1", "0.0.0.0") or host.startswith("127.")


def _default_scheme(host: str) -> str:
    # HTTPS by default, HTTP only for local dev.
    return "http" if _is_local_host(host) else "https"


# Provider-specific defaults.
# Key: host suffix match
# Value: default base_path
_PROVIDER_BASE_PATH = {
    # Volcengine Ark (Doubao)
    "volces.com": "/api/v3",
    "volcengineapi.com": "/api/v3",
    "bytepluses.com": "/api/v3",
    # MiniMax OpenAI-compatible API
    "api.minimaxi.com": "/v1",
}


_RESERVED_QS_KEYS = {
    "api_key",
    "api_key_env",
    "base_url",
    "base_path",
    "timeout",
    "verify",
    "insecure",
    "ca_file",
    "ca_path",
}


def _default_base_path(host: str) -> str:
    for suffix, path in _PROVIDER_BASE_PATH.items():
        if host.endswith(suffix):
            return path
    return "/v1"


def _provider_name_from_url(u: urllib.parse.ParseResult) -> str:
    scheme = u.scheme or ""
    if scheme.startswith("llm+"):
        provider = scheme.split("+", 1)[1].strip()
        if provider:
            return provider
    return "openai"


def openai_compat_factory(u: urllib.parse.ParseResult, qs: Dict[str, str]) -> LLMAdapter:
    model = (u.path or "").lstrip("/")
    if not model:
        raise ValueError("Model missing in DSN path, e.g. llm+openai://host/MODEL?....")

    provider_name = _provider_name_from_url(u)
    host = u.netloc.strip()
    base_url = qs.get("base_url")
    if base_url:
        base = base_url.rstrip("/")
    else:
        if host:
            if host == "api.openai.com":
                base = "https://api.openai.com"
            elif host == "api.minimaxi.com":
                base = "https://api.minimaxi.com"
            else:
                scheme = _default_scheme(host)
                base = f"{scheme}://{host}"
        else:
            if provider_name == "minimax":
                base = "https://api.minimaxi.com"
                host = "api.minimaxi.com"
            else:
                base = "https://api.openai.com"

    base_path = qs.get("base_path", _default_base_path(host)).rstrip("/")
    endpoint = f"{base}{base_path}/chat/completions"

    api_key = qs.get("api_key")
    if not api_key:
        env = qs.get("api_key_env")
        if env:
            api_key = os.environ.get(env, "")
    if api_key == "none":
        api_key = ""

    timeout = _maybe_float(qs.get("timeout"))
    payload_params = {
        k: _parse_payload_value(v)
        for k, v in qs.items()
        if k not in _RESERVED_QS_KEYS
    }

    # TLS options from DSN ---
    # verify: default True. insecure=1 is shorthand for verify=False.
    verify_q = (qs.get("verify") or "").strip().lower()
    insecure_q = (qs.get("insecure") or "").strip().lower()

    verify = True
    if verify_q in ("0", "false", "no", "off"):
        verify = False
    if insecure_q in ("1", "true", "yes", "on"):
        verify = False

    tls = TLSConfig(
        verify=verify,
        ca_file=qs.get("ca_file") or None,
        ca_path=qs.get("ca_path") or None,
    )


    def _headers() -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if api_key:
            h["Authorization"] = f"Bearer {api_key}"
        return h

    def _body(prompt: str, stream: bool, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(payload_params)
        payload.update({
            "model": model,
            "messages": _normalize_messages(prompt, messages),
            "stream": stream,
        })
        return payload

    async def _post_json(payload):
        def _do():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=_headers(), method="POST")
            try:
                with urlopen_with_tls(req, timeout=timeout, tls=tls) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                raise_http_error(
                    e,
                    provider=provider_name,
                    hint="Check API key, model name, and endpoint."
                )
            except urllib.error.URLError as e:
                if is_timeout_error(e):
                    raise_timeout_error(
                        e,
                        provider=provider_name,
                        hint="Increase DSN timeout or Agent timeout for slow models."
                    )
                raise_url_error(
                    e,
                    provider=provider_name,
                    hint="Check network and base_url/host."
                )
            except Exception as e:
                if is_timeout_error(e):
                    raise_timeout_error(
                        e,
                        provider=provider_name,
                        hint="Increase DSN timeout or Agent timeout for slow models."
                    )
                raise

        return await run_blocking(_do)

    async def _post_stream(payload: Dict[str, Any]) -> AsyncGenerator[str, None]:
        def _open():
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(endpoint, data=data, headers=_headers(), method="POST")
            return urlopen_with_tls(req, timeout=timeout, tls=tls)

        # ---- open connection (already handled) ----
        try:
            resp = await run_blocking(_open)
        except urllib.error.HTTPError as e:
            raise_http_error(
                e,
                provider=provider_name,
                hint="Check API key, model name, and endpoint."
            )
        except urllib.error.URLError as e:
            if is_timeout_error(e):
                raise_timeout_error(
                    e,
                    provider=provider_name,
                    hint="Increase DSN timeout or Agent timeout for slow models."
                )
            raise_url_error(
                e,
                provider=provider_name,
                hint="Check network and base_url/host."
            )
        except Exception as e:
            if is_timeout_error(e):
                raise_timeout_error(
                    e,
                    provider=provider_name,
                    hint="Increase DSN timeout or Agent timeout for slow models."
                )
            raise

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
                provider=provider_name,
                hint="Stream interrupted; check API key/model/endpoint."
            )
        except urllib.error.URLError as e:
            if is_timeout_error(e):
                raise_timeout_error(
                    e,
                    provider=provider_name,
                    hint="Increase DSN timeout or Agent timeout for slow models."
                )
            raise_url_error(
                e,
                provider=provider_name,
                hint="Stream interrupted; check network and base_url/host."
            )
        except Exception as e:
            if is_timeout_error(e):
                raise_timeout_error(
                    e,
                    provider=provider_name,
                    hint="Increase DSN timeout or Agent timeout for slow models."
                )
            raise RuntimeError(f"[{provider_name}] Stream interrupted: {e}") from None

        finally:
            try:
                resp.close()
            except Exception:
                pass

    def adapter(prompt: str, stream: bool, *, messages: Optional[List[Dict[str, Any]]] = None):
        if stream:
            return _post_stream(_body(prompt, True, messages=messages))

        async def _one() -> str:
            obj = await _post_json(_body(prompt, False, messages=messages))
            return _response_to_result(obj)

        return _one()

    return adapter


# register aliases
register_llm_provider("openai", openai_compat_factory)
register_llm_provider("openai_compat", openai_compat_factory)
register_llm_provider("minimax", openai_compat_factory)
