import os
import asyncio
from picoflow import flow, llm, create_agent, State
from picoflow.adapters import from_url

llm_adapter = from_url(os.environ["LLM_DSN"])


# -----------------------
# Tool
# -----------------------

def search_information(query: str) -> str:
    print(f"\n--- 🛠️ 工具调用：search_information, 查询：'{query}' ---")
    simulated_results = {
        "weather in london": "伦敦当前天气多云，气温 15°C。",
        "capital of france": "法国的首都是巴黎。",
        "population of earth": "地球人口约 80 亿。",
        "tallest mountain": "珠穆朗玛峰是海拔最高的山峰。",
        "default": f"模拟搜索 '{query}'：未找到具体信息，但该主题很有趣。"
    }
    result = simulated_results.get(query.lower(), simulated_results["default"])
    print(f"--- 工具结果：{result} ---")
    return result


TOOLS = {
    "search_information": search_information
}


# -----------------------
# LLM (Planner)
# -----------------------

decide = llm(
    "你是一个乐于助人的助手，并且你可以使用一个工具：search_information(query)。\n"
    "当你需要事实信息（如首都、天气、人口、最高山等）时，必须先调用工具再回答。\n"
    "否则直接回答。\n\n"
    "如果需要调用工具，请直接调用，不要自行编造结果。\n"
    "最终请直接给出回答内容。\n\n"
    "用户问题：{input}\n"
    "{memory}\n",
    llm_adapter=llm_adapter,
    final=True,
)


# -----------------------
# Seed
# -----------------------

@flow
def seed(ctx: State) -> State:
    q = ctx.input
    ctx = ctx.add_memory("user", q)
    return ctx.update(done=False)


# -----------------------
# Agent
# -----------------------

from picoflow import tool_loop

agent = create_agent(
    seed >> tool_loop(
        decide,
        tools=TOOLS,
        max_steps=6,
    )
)


# -----------------------
# Run
# -----------------------

async def run_query(q: str):
    print(f"\n--- 🏃 Agent 运行查询：'{q}' ---")
    s = await agent.arun(q)
    print("\n--- ✅ Agent 最终回复 ---")
    print(s.output)


async def main():
    await asyncio.gather(
        run_query("What is the capital of France?"),
        run_query("What's the weather like in London?"),
        run_query("Tell me something about dogs."),
    )


if __name__ == "__main__":
    asyncio.run(main())
