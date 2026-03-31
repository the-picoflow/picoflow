# Timeout Repro

This cookbook starts a local mock OpenAI-compatible server and reproduces three
timeout scenarios without calling a real model.

## Run

Agent timeout wins:

```bash
python3 picoflow/cookbook/timeout_repro/main.py --scenario agent-timeout
```

Expected result:

```text
TimeoutError: Agent execution timed out
```

Request timeout wins for non-streaming:

```bash
python3 picoflow/cookbook/timeout_repro/main.py --scenario request-timeout
```

Expected result:

```text
RuntimeError: [openai] Request timed out: ...
```

Request timeout wins while streaming:

```bash
python3 picoflow/cookbook/timeout_repro/main.py --scenario stream-request-timeout
```

Expected result:

```text
RuntimeError: [openai] Request timed out: ...
```

## Notes

- The script binds a local server on `127.0.0.1` with a random free port.
- It uses `llm+openai://127.0.0.1:<port>/mock-model?api_key=none&timeout=...`.
- You can override timing knobs with `--agent-timeout`, `--request-timeout`,
  `--response-delay`, and `--stream-chunk-delay`.
