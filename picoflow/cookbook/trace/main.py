import os
import asyncio
from picoflow import Flow, State, llm, create_agent
from picoflow.adapters.registry import from_url

# ---- tracer ----
def tracer(event, data: dict) -> None:
    # Minimal structured trace output
    print(f"[TRACE] {event.value} | {data}")

# ---- flows ----
async def prep(ctx: State) -> State:
    # Put user input into a simple promptable form
    return ctx.add_memory("user", ctx.input)

# Real adapter via DSN (adjust to your env / provider)
DSN = os.environ.get(
    "LLM_DSN",
    "llm+openai://api.siliconflow.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B?api_key_env=OPENAI_API_KEY",
)
adapter = from_url(DSN)

# DSL: compose flows (prep -> llm)
flow = (
    Flow(prep, "prep")
    >> llm(
        prompt_template="You are a helpful assistant.\n\nUser: {input}\n",
        llm_adapter=adapter,
        final=True,
    )
)

agent = create_agent(flow)

async def main():
    state = await agent.arun("Explain trace in one sentence.", trace=tracer, timeout=30)
    print("RESULT:", state.output)

if __name__ == "__main__":
    asyncio.run(main())
