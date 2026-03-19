# FastAPI Minimal

This example shows how to use PicoFlow with the current `openai_compat` adapter in a FastAPI app.

## Install

```bash
pip install -e .
pip install fastapi uvicorn
```

## Configure

Set your API key and optionally override the DSN:

```bash
export OPENAI_API_KEY=your_api_key
export LLM_DSN='llm+openai://ark.cn-beijing.volces.com/your_doubao_model?api_key_env=OPENAI_API_KEY&timeout=300'
```

Notes:

- `timeout=300` is recommended for Doubao streaming to avoid premature socket timeout.
- If your endpoint or certificate chain is special, you can add `base_path`, `verify`, `insecure`, `ca_file`, or `ca_path` in the DSN.

## Run

```bash
uvicorn picoflow.cookbook.fastapi_minimal.main:app --reload
```

## Try it

Non-streaming:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"用一句话介绍豆包模型。"}'
```

Streaming (SSE):

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"请逐步解释什么是 FastAPI。","timeout":300}'
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```
