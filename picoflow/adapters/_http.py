# adapters/_http.py
import json
import urllib.error
import urllib.request
import asyncio
from typing import Callable, TypeVar, Optional
import os
import ssl
from dataclasses import dataclass

T = TypeVar("T")


async def run_blocking(fn: Callable[[], T]) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn)


# --- PATCH BEGIN: TLSConfig + urlopen_with_tls ---

@dataclass(frozen=True)
class TLSConfig:
    """
    Transport-level TLS options (applies to all providers uniformly).

    Priority (highest -> lowest):
      1) DSN query params (verify/insecure/ca_file/ca_path)
      2) PICO_* env vars
      3) SSL_CERT_* env vars
    """
    verify: bool = True
    ca_file: Optional[str] = None
    ca_path: Optional[str] = None


def _env_bool(name: str) -> Optional[bool]:
    v = os.environ.get(name)
    if v is None or v == "":
        return None
    v = v.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def _resolve_tls_config(tls: Optional[TLSConfig]) -> TLSConfig:
    # Start from provided (DSN-level), then fill missing from envs.
    verify = tls.verify if tls is not None else True
    ca_file = tls.ca_file if tls is not None else None
    ca_path = tls.ca_path if tls is not None else None

    # DSN didn't set explicit CA: allow env fallback
    if ca_file is None:
        ca_file = os.environ.get("PICO_CA_FILE") or os.environ.get("SSL_CERT_FILE") or None
    if ca_path is None:
        ca_path = os.environ.get("PICO_CA_PATH") or os.environ.get("SSL_CERT_PATH") or None

    # DSN didn't override verify? allow env override
    env_verify = _env_bool("PICO_SSL_VERIFY")
    if env_verify is not None:
        verify = env_verify

    return TLSConfig(verify=verify, ca_file=ca_file, ca_path=ca_path)


def build_ssl_context(tls: Optional[TLSConfig]) -> Optional["ssl.SSLContext"]:
    cfg = _resolve_tls_config(tls)

    if not cfg.verify:
        # Insecure: no certificate verification (debug only).
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    # verify = True
    # If ca_file/ca_path provided, build context with them.
    if cfg.ca_file or cfg.ca_path:
        return ssl.create_default_context(cafile=cfg.ca_file, capath=cfg.ca_path)

    # Otherwise use default system/certifi chain as available.
    return ssl.create_default_context()


def urlopen_with_tls(req: urllib.request.Request, timeout: Optional[float], tls: Optional[TLSConfig]):
    """
    Centralized urllib urlopen with TLS handling.
    Adapters should call this instead of urllib.request.urlopen directly.
    """
    ctx = build_ssl_context(tls)
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)


# --- PATCH END ---


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
