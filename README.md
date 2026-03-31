<p align="center">
  <img src="picoflow/assets/picoflow_logo.png" width="280">
</p>
<p align="center">
  <img src="https://img.shields.io/pypi/dm/picoflow">  <img src="https://img.shields.io/pypi/l/picoflow">
</p>



# PicoFlow — Simple, Flexible AI Agent Framework

**Build agents with explicit steps and a small DSL.  
LLMs, tools, loops, and branches compose naturally.**

---

## A Minimal PicoFlow Application

```python
from picoflow import flow, llm, create_agent

LLM_URL = "llm+openai:///gpt-4.1-mini?api_key_env=OPENAI_API_KEY"

@flow
async def mem(ctx):
    return ctx.add_memory("user", ctx.input)

agent = create_agent(
    mem >> llm("Answer in one sentence: {input}", llm_adapter=LLM_URL)
)

print(agent.get_output("What is PicoFlow?", trace=True))
```

```bash
export OPENAI_API_KEY=sk-...
python minimal.py
```

---

## Core Ideas

- **Flow = step**  
  A flow is just a Python function that takes and returns `State`.

- **DSL = pipeline**  
  Use `>>` to compose steps into readable execution graphs.

- **Agent = runner**  
  `create_agent(flow)` gives you `run / arun / get_output`.

- **State = context (Ctx)**  
  `Ctx` is an alias of `State`. It is immutable and explicit.

---

## Quick Start (Step by Step)

### 1. Define Steps with `@flow`

```python
from picoflow import flow, Ctx

@flow
async def normalize(ctx: Ctx) -> Ctx:
    return ctx.update(input=ctx.input.strip().lower())
```

---

### 2. Call LLM as a Step

```python
from picoflow import llm

ask = llm("Answer briefly: {input}")
```

---

### 3. Compose with DSL

```python
pipeline = normalize >> ask
```

---

### 4. Run with Agent

```python
from picoflow import create_agent

agent = create_agent(pipeline)

state = await agent.arun("Hello WORLD")
print(state.output)
```

---

## DSL in One Minute

### Sequential

```python
flow = a >> b >> c
```

### Loop

```python
flow = step.repeat()
```

or:

```python
flow = repeat(step, until=lambda s: s.done)
```

### Parallel + Merge

```python
flow = fork(a, b) >> merge()
```

Custom merge:

```python
flow = fork(a, b) >> merge(
    mode=MergeType.CUSTOM,
    reducer=lambda branches, main: branches[0]
)
```

---

## LLM URL

```python
from picoflow.adapters.registry import from_url

adapter = from_url(
    "llm+openai:///gpt-4.1-mini?api_key_env=OPENAI_API_KEY"
)
```

Then:

```python
flow = llm("Explain: {input}", llm_adapter=adapter)
```

### Custom Adapters

```python
class MyAdapter(LLMAdapter):
    def __call__(self, prompt: str, stream: bool):
        ...
```

```python
from picoflow.adapters.registry import register

register("myllm", lambda url: MyAdapter(...))
```

Use:

```
llm+myllm://host/model?param=value
```

### OpenAI-Compatible DSN Notes

The OpenAI-compatible adapter uses this shape:

```text
llm+openai://<host>/<model>?k=v&...
```

Examples:

```text
llm+openai:///gpt-4.1-mini?api_key_env=OPENAI_API_KEY
llm+minimax:///MiniMax-Text-01?api_key_env=MINIMAX_API_KEY
llm+openai://ark.cn-beijing.volces.com/doubao-seed-1-8-251228?api_key_env=OPENAI_API_KEY
llm+minimax://api.minimaxi.com/MiniMax-Text-01?api_key_env=MINIMAX_API_KEY
llm+openai://127.0.0.1:8000/Qwen2.5?api_key=none
```

Supported standard query params include:

- `api_key`
- `api_key_env`
- `base_url`
- `base_path`
- `timeout`
- `verify`
- `insecure`
- `ca_file`
- `ca_path`

All other query params are passed through to the request payload
automatically.

Example: Doubao `thinking`

```text
llm+openai://ark.cn-beijing.volces.com/doubao-seed-1-8-251228?api_key_env=OPENAI_API_KEY&thinking={"type":"enabled"}
```

Example: OpenAI-compatible reasoning parameter

```text
llm+openai:///gpt-5.1?api_key_env=OPENAI_API_KEY&reasoning_effort=medium
```

Example: MiniMax via the OpenAI-compatible adapter alias

```text
llm+minimax:///MiniMax-Text-01?api_key_env=MINIMAX_API_KEY
```

Simple JSON values are parsed automatically:

- JSON objects and arrays become dict/list
- JSON numbers become int/float
- `true` / `false` / `null` become boolean / `None`
- other values stay as strings

Core fields such as `model`, `messages`, and `stream` are still controlled by
PicoFlow and are not taken from the query string.

---

## Runtime Options

### Tracing

```python
await agent.arun("hi", trace=True)
```

### Timeout

```python
await agent.arun("hi", timeout=10)
```

### Streaming

```python
async def on_chunk(text: str):
    print(text, end="", flush=True)

await agent.arun("stream me", stream_callback=on_chunk)
```

---

## Tools

```python
from picoflow import tool

flow = tool("search", lambda q: {"result": "..."} )
```

Results:

```python
state.tools["search"]
```

## Troubleshooting: SSL certificate verify failed

**Recommended (secure):** set your CA bundle

``` bash
export SSL_CERT_FILE=/path/to/ca.pem
# or
export PICO_CA_FILE=/path/to/ca.pem
```

**Quick debug (insecure):** disable verification temporarily

``` text
...&insecure=1
```

or

``` bash
export PICO_SSL_VERIFY=0
```

**Do not use this in production.**

Note: Local Ollama usage (`llm+ollama://localhost:11434/...`) uses plain
HTTP and does not require SSL configuration.


---
## 📚 Cookbook (Examples)

The `cookbook/` directory contains small, focused examples for common
patterns.\
Each folder is runnable and demonstrates one specific feature or usage
style.

    cookbook/
    ├─ minimal-demo        # Smallest runnable PicoFlow example
    ├─ multiple-crew       # Multi-agent / crew-style collaboration
    ├─ simple-chat         # Basic chat agent
    ├─ simple-chat-stream  # Streaming responses from LLM
    ├─ simple-tool         # Tool calling and tool state
    └─ trace               # Tracing and execution visualization

### How to run

All examples are standalone scripts:

``` bash
export OPENAI_API_KEY=sk-...
cd cookbook/simple-chat
python main.py
```

### When to use which example

-   **Start here** → `minimal-demo`\
    Understand Flow, DSL (`>>`), and Agent execution.

-   **LLM interaction** → `simple-chat`, `simple-chat-stream`\
    Prompt → response, with and without streaming.

-   **Tool calling** → `simple-tool`\
    How tools are defined, executed, and stored in `state.tools`.

-   **Multi-step / multi-agent** → `multiple-crew`\
    Composition of multiple flows and roles.

-   **Debugging & observability** → `trace`\
    How tracing hooks and structured events work.

> Cookbook examples are intentionally small and close to real usage.\
> They are recommended as the primary learning path after reading the
> Minimal Example.
---
## License

MIT
