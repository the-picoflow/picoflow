from picoflow import flow, llm, create_agent

# Connect model with LLM URL (API key from env: OPENAI_API_KEY)
LLM_URL = "llm+openai://api.siliconflow.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B?api_key_env=OPENAI_API_KEY"

@flow
async def mem(ctx):
    # Put user input into memory
    return ctx.add_memory("user", ctx.input)

# Compose pipeline with DSL and run
agent = create_agent(
    mem >> llm("Answer in one sentence: {input}", llm_adapter=LLM_URL)
)

print(agent.get_output("What is openai?", trace=True))
