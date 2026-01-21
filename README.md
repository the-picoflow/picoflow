<p align="center">
  <img src="picoflow/assets/picoflow_logo.png" width="280">
</p>

# PicoFlow — Simple, Flexible AI Agent Framework

**Build agents with explicit steps and a small DSL.  
LLMs, tools, loops, and branches compose naturally.**

---

## A Minimal PicoFlow Application

```python
from picoflow import flow, llm, create_agent

LLM_URL = "llm+openai://api.openai.com/v1/chat/completions?model=gpt-4.1-mini&api_key_env=OPENAI_API_KEY"

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
    "llm+openai://api.openai.com/v1/chat/completions"
    "?model=gpt-4.1-mini&api_key_env=OPENAI_API_KEY"
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

---

## License

MIT
