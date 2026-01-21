import os
import inspect
from picoflow import flow, Flow, State
from picoflow.adapters import from_url

DSN = os.environ.get("LLM_DSN",
                     "llm+openai://api.siliconflow.cn/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B?api_key_env=OPENAI_API_KEY")
llm_adapter = from_url(DSN)


@flow
def init(ctx):
    ctx.metadata.setdefault("messages", [])
    if not ctx.metadata["messages"]:
        print(f"Using: {DSN}")
        print("Welcome to the chat! Type 'exit' to end the conversation.")
    return ctx


@flow
def read_or_exit(ctx):
    text = input("\nYou: ").strip()
    if text.lower() == "exit":
        print("\nGoodbye!")
        return ctx.update(done=True, stop_reason="exit")
    return ctx.update(input=text)


@flow
def remember_user(ctx):
    ctx.metadata["messages"].append({"role": "user", "content": ctx.input})
    return ctx


@flow
async def ask_llm(ctx):
    # build prompt inline (no helper)
    prompt = "\n".join(
        f"{m['role']}: {m['content']}" for m in ctx.metadata["messages"]
    ) + "\nassistant:"

    v = llm_adapter(prompt, False)  # stream=False
    out = await v if inspect.isawaitable(v) else v
    return ctx.update(output=out)


@flow
def show_and_remember(ctx):
    print(f"\nAssistant: {ctx.output}")
    ctx.metadata["messages"].append({"role": "assistant", "content": ctx.output})
    return ctx


chat = (
        init
        >> read_or_exit
        >> remember_user
        >> ask_llm
        >> show_and_remember
).repeat()

if __name__ == "__main__":
    chat.run(State(metadata={}))
