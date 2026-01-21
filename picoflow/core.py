from dataclasses import dataclass, replace, field
from functools import wraps
from typing import Callable, Optional, Dict, Any, List, Union, Generator, Awaitable, AsyncGenerator, AsyncIterator, \
    Iterator
from enum import Enum, auto
import time
import inspect
import asyncio
from .adapters.types import LLMAdapter
from .adapters.registry import from_url


class TraceEvent(str, Enum):
    FLOW_START = "flow.start"
    FLOW_END = "flow.end"
    FLOW_ERROR = "flow.error"


Tracer = Callable[[TraceEvent, Dict[str, Any]], None]


def default_tracer(event: TraceEvent, data: Dict[str, Any]) -> None:
    print(f"[TRACE] {event.value} | {data}")


class Flow:
    def __init__(self, fn: Union[Callable[["State"], "State"], Callable[["State"], Awaitable["State"]]],
                 name: Optional[str] = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", fn.__class__.__name__)

    def __call__(self, state: "State") -> "State":
        raise TypeError(
            f"Flow '{self.name}' is async-only; use 'await flow.acall(state)' (or 'await agent.arun(...)')."
        )

    async def acall(self, state: "State") -> "State":
        result = self.fn(state)

        if inspect.isawaitable(result):
            result = await result

        if not isinstance(result, State):
            raise TypeError(
                f"Flow '{self.name}' must return State, got {type(result).__name__}"
            )
        return result

    def __rshift__(self, other: "Flow") -> "Flow":
        return pipe(self, other)

    def run(self, state: "State") -> "State":
        return run_awaitable(self.acall(state))

    async def run_async(self, state: "State") -> "State":
        return await self.acall(state)

    def repeat(self) -> "Flow":
        return repeat(self)


def flow(fn: Callable[["State"], Union["State", Awaitable["State"]]]) -> Flow:
    @wraps(fn)
    def wrapped(state: State):
        return fn(state)

    return Flow(wrapped, name=fn.__name__)


@dataclass(frozen=True)
class State:
    input: str = ""
    output: str = ""
    memory: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    tools: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    done: bool = False
    stop_reason: Optional[str] = None
    branches: Optional[List["State"]] = None

    def update(self, **kwargs) -> "State":
        return replace(self, **kwargs)

    def add_memory(self, role: str, content: str) -> "State":
        return self.update(memory=self.memory
                                  + [{"role": role,
                                      "content": content,
                                      "time": time.time()}])

    def with_data(self, **kv) -> "State":
        # copy-on-write: keep "logical immutability"
        return replace(self, data={**self.data, **kv})

Ctx = State


def pipe(*flows: Flow) -> Flow:
    async def run(state: State) -> State:
        current = state
        for f in flows:
            check_timeout(current)
            if current.done:
                return current

            if current.branches is not None:
                raise RuntimeError(
                    "Encountered branches without merge; "
                    "please add merge() before continuing pipeline."
                )

            current = await f.acall(current)
            check_timeout(current)

            if current.done:
                return current

        return current

    run.__flows__ = flows
    return Flow(run, name="pipe")


def fork(*flows: Flow) -> Flow:
    async def run(state: State) -> State:
        async def one(i: int, f: Flow) -> State:
            child = state.update(metadata={**state.metadata, "fork_id": i})
            return await f.acall(child)

        branches = await asyncio.gather(*(one(i, f) for i, f in enumerate(flows)))
        return state.update(branches=list(branches))

    return Flow(run, name="fork")


class MergeType(Enum):
    CONCAT_OUTPUT = auto()
    PICK_FIRST = auto()
    PICK_LAST = auto()
    CUSTOM = auto()


def merge(
        mode: MergeType = MergeType.CONCAT_OUTPUT,
        *,
        reducer: Optional[Callable[[List[State], State], State]] = None,
) -> Flow:
    def run(state: State) -> State:
        if not state.branches:
            raise RuntimeError("merge() called but no branches found")

        def merge_main_metadata(picked: State, main: State) -> Dict[str, Any]:
            keys = ("deadline", "stream_callback", "trace")
            merged = dict(picked.metadata)
            for k in keys:
                if k in main.metadata:
                    merged[k] = main.metadata[k]
            return merged

        branches = state.branches

        if mode == MergeType.CONCAT_OUTPUT:
            output = "\n".join(s.output for s in branches if s.output)
            base = state.memory
            base_len = len(base)
            merged_memory = base
            for s in branches:
                if len(s.memory) >= base_len and s.memory[:base_len] == base:
                    merged_memory = merged_memory + s.memory[base_len:]
                else:
                    merged_memory = merged_memory + [m for m in s.memory if m not in base]

            merged = state.update(output=output, memory=merged_memory)

        elif mode == MergeType.PICK_FIRST:
            picked = branches[0]
            merged = picked.update(
                metadata=merge_main_metadata(picked, state),
                branches=None,
            )

        elif mode == MergeType.PICK_LAST:
            picked = branches[-1]
            merged = picked.update(
                metadata=merge_main_metadata(picked, state),
                branches=None,
            )

        elif mode == MergeType.CUSTOM:
            if reducer is None:
                raise ValueError("CUSTOM merge requires reducer")
            merged = reducer(branches, state)
            if not isinstance(merged, State):
                raise TypeError(f"merge CUSTOM reducer must return State, got {type(merged).__name__}")

        else:
            raise ValueError(f"Unsupported merge mode: {mode}")

        return merged.update(branches=None)

    return Flow(run, name=f"merge:{mode.name.lower()}")


def tool(name: str, fn: Callable[[Any], Any]) -> Flow:
    async def run(state: State) -> State:
        try:
            result = fn(state.input)
            if inspect.isawaitable(result):
                result = await result
            return state.update(tools={**state.tools, name: result})
        except Exception as e:
            return state.update(tools={**state.tools, name: {"error": str(e)}})

    return Flow(run, name=f"tool:{name}")


def llm(
        prompt_template: str = "{input}",
        stream: bool = False,
        llm_adapter: Optional[Union[str, LLMAdapter]] = None,
        final: bool = True,
) -> Flow:
    def default_adapter(prompt: str, _stream: bool):
        if _stream:
            def gen():
                yield f"[chunk] {prompt}"

            return gen()
        return prompt

    if isinstance(llm_adapter, str):
        adapter = from_url(llm_adapter)
    else:
        adapter = llm_adapter or default_adapter

    async def _to_async_iter(
            v: Union[
                str,
                Awaitable[str],
                Generator[str, None, None],
                AsyncGenerator[str, None],
                Awaitable[Generator[str, None, None]],
                Awaitable[AsyncGenerator[str, None]],
            ]
    ) -> AsyncIterator[str]:
        """
        Normalize different adapter return types into an async iterator of strings.

        Supported forms:
        - str
        - Awaitable[str]
        - Generator[str]
        - AsyncGenerator[str]
        - Awaitable[Generator[str] or AsyncGenerator[str]]
        """

        # Some SDKs require awaiting before returning the stream iterator
        if inspect.isawaitable(v):
            v = await v  # type: ignore[misc]

        # Async generator: consume with async for
        if inspect.isasyncgen(v) or hasattr(v, "__aiter__"):
            async for c in v:  # type: ignore[union-attr]
                if not isinstance(c, str):
                    raise TypeError(f"llm stream chunk must be str, got {type(c).__name__}")
                yield c
            return

        # Single string: treat as one chunk
        if isinstance(v, str):
            yield v
            return

        # Sync generator or iterable of strings
        for c in v:  # type: ignore[union-attr]
            if not isinstance(c, str):
                raise TypeError(f"llm stream chunk must be str, got {type(c).__name__}")
            yield c

    async def run(state: State) -> State:
        # Build prompt from input and memory
        memory_str = "\n".join(f"{m['role']}: {m['content']}" for m in state.memory)
        evidence_block = ""
        if state.evidence:
            evidence_block = "\n\nEvidence:\n" + "\n".join(f"- {t}" for t in state.evidence if isinstance(t, str) and t)

        prompt = prompt_template.format(
            input=state.input,
            memory=memory_str,
            evidence=evidence_block,
        )

        if "{evidence}" not in prompt_template:
            prompt = prompt + evidence_block

        # -------- streaming mode --------
        if stream:
            cb = state.metadata.get("stream_callback")
            if cb is None:
                raise ValueError(
                    "llm(stream=True) requires stream_callback via Agent.arun(..., stream_callback=...)"
                )

            chunks: List[str] = []
            result = adapter(prompt, True)

            async for c in _to_async_iter(result):
                chunks.append(c)
                r = cb(c)
                if inspect.isawaitable(r):
                    await r

            output = "".join(chunks)

        # -------- non-streaming mode --------
        else:
            result = adapter(prompt, False)

            if inspect.isawaitable(result):
                output = await result  # type: ignore[misc]
            else:
                output = result  # type: ignore[assignment]

            if not isinstance(output, str):
                raise TypeError(
                    f"llm(stream=False) adapter must return str or Awaitable[str], got {type(output).__name__}"
                )

        return state.update(
            output=output,
            memory=state.add_memory("assistant", output).memory,
            done=True,
            stop_reason="final" if final else None,
        )

    return Flow(run, name="llm")


def check_timeout(state: State):
    deadline = state.metadata.get("deadline")
    if deadline is not None and time.time() > float(deadline):
        raise TimeoutError("Agent execution timed out")


def wrap_timeout(flow: Flow) -> Flow:
    base_flow = flow

    async def run_with_timeout(state: State) -> State:
        check_timeout(state)
        result = await base_flow.acall(state)
        check_timeout(result)
        return result

    return Flow(run_with_timeout, name=f"{flow.name}:timeout")


def wrap_trace(flow: Flow, tracer: Optional[Tracer]) -> Flow:
    if tracer is None:
        return flow

    base_flow = flow

    async def traced(state: State) -> State:
        tracer(TraceEvent.FLOW_START, {"input": state.input})
        try:
            result = await base_flow.acall(state)
        except Exception as e:
            tracer(TraceEvent.FLOW_ERROR, {"error": str(e)})
            raise
        tracer(TraceEvent.FLOW_END, {
            "output": result.output,
            "done": result.done,
            "stop_reason": result.stop_reason,
            "evidence_count": len(getattr(result, "evidence", []) or []),
        })
        return result

    return Flow(traced, name=f"{flow.name}:trace")


def resolve_tracer(trace: Union[None, bool, Tracer]) -> Optional[Tracer]:
    if trace is None or trace is False:
        return None
    elif trace is True:
        def print_tracer(event: TraceEvent, info: Dict[str, Any]):
            print(f"[TRACE] {event.value}: {info}")

        return print_tracer
    elif callable(trace):
        return trace
    else:
        raise ValueError("trace must be None, bool, or callable")


class Agent:

    def __init__(self, flow: Flow):
        self.flow = flow

    async def arun(
            self,
            input: str,
            *,
            stream_callback: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None,
            trace: Union[None, bool, Tracer] = False,
            timeout: Optional[float] = None
    ) -> State:
        tracer = resolve_tracer(trace)

        metadata: Dict[str, Any] = {}
        if stream_callback:
            metadata["stream_callback"] = stream_callback
        if trace:
            metadata["trace"] = True
        if timeout is not None:
            metadata["deadline"] = time.time() + timeout

        state = State(input=input, metadata=metadata)

        f = self.flow
        f = wrap_timeout(f)
        f = wrap_trace(f, tracer)

        return await f.acall(state)

    def run(self, *args, **kwargs) -> State:
        return run_awaitable(self.arun(*args, **kwargs))

    def get_output(self, input: str, **kwargs) -> str:
        return self.run(input, **kwargs).output

    async def aget_output(self, input: str, **kwargs) -> str:
        return (await self.arun(input, **kwargs)).output


def repeat(step: Flow, until: Optional[Callable[["State"], bool]] = None) -> Flow:
    async def run(ctx: State) -> State:
        while True:
            if until is None:
                if ctx.done:
                    return ctx
            else:
                if until(ctx):
                    return ctx
            ctx = await step.acall(ctx)
    return Flow(run, name=f"repeat({step.name})")


def create_agent(flow: Flow) -> Agent:
    return Agent(flow)


def run_awaitable(awaitable):
    """
    Run an awaitable from sync code.

    - If no event loop is running: uses asyncio.run()
    - If called inside a running event loop: raise (user must 'await')
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError(
        "Cannot run sync wrapper inside a running event loop; use 'await ...' instead."
    )
