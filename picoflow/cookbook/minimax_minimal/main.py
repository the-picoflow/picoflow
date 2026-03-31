import os

from picoflow import create_agent, flow, llm


LLM_DSN = os.environ.get(
    "LLM_DSN",
    "llm+minimax:///MiniMax-M2.7?api_key_env=MINIMAX_API_KEY&timeout=300&insecure=1&reasoning_split=true",
)


@flow
async def remember_user(ctx):
    return ctx.add_memory("user", ctx.input)


agent = create_agent(
    remember_user >> llm("Answer briefly: {input}", llm_adapter=LLM_DSN)
)


if __name__ == "__main__":
    print(f"Using: {LLM_DSN}")
    print(agent.get_output("Introduce MiniMax in one sentence.", trace=True))
