# MiniMax Minimal

This is the smallest PicoFlow example for the `minimax` provider alias.

## Run

```bash
export MINIMAX_API_KEY=your_api_key
python3 picoflow/cookbook/minimax_minimal/main.py
```

## TLS note

The default example DSN includes `insecure=1`:

```text
llm+minimax:///MiniMax-M2.7?api_key_env=MINIMAX_API_KEY&timeout=300&insecure=1&reasoning_split=true
```

This is only to make the minimal example easier to run in environments with
custom proxy or self-signed TLS chains.

If your machine has a valid CA chain, prefer removing `insecure=1`.

If your environment uses a custom CA, prefer one of these instead:

```bash
export LLM_DSN='llm+minimax:///MiniMax-M2.7?api_key_env=MINIMAX_API_KEY&timeout=300&ca_file=/path/to/ca.pem&reasoning_split=true'
```

or:

```bash
export PICO_CA_FILE=/path/to/ca.pem
python3 picoflow/cookbook/minimax_minimal/main.py
```
