import asyncio
import os
from contextlib import suppress
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from picoflow import create_agent, llm
import uvicorn


LLM_DSN = os.environ.get(
    "LLM_DSN",
    "llm+openai://ark.cn-beijing.volces.com/doubao-seed-1-8-251228?api_key_env=OPENAI_API_KEY&insecure=1",
)


app = FastAPI(title="PicoFlow FastAPI Minimal")

chat_agent = create_agent(
    llm(
        "You are a concise assistant.\n\nUser: {input}\nAssistant:",
        llm_adapter=LLM_DSN,
    )
)

stream_agent = create_agent(
    llm(
        "You are a concise assistant.\n\nUser: {input}\nAssistant:",
        stream=True,
        llm_adapter=LLM_DSN,
    )
)


class ChatRequest(BaseModel):
    message: str
    timeout: Optional[float] = 60


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    state = await chat_agent.arun(req.message, timeout=req.timeout)
    return {"output": state.output}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    queue: "asyncio.Queue[Optional[str]]" = asyncio.Queue()

    async def on_chunk(chunk: str) -> None:
        await queue.put(chunk)

    async def run_agent() -> None:
        try:
            await stream_agent.arun(
                req.message,
                timeout=req.timeout,
                stream_callback=on_chunk,
            )
        except Exception as e:
            await queue.put(f"\n[ERROR] {e}")
        finally:
            await queue.put(None)

    async def event_gen():
        task = asyncio.create_task(run_agent())
        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                yield f"data: {chunk}\n\n"
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    return StreamingResponse(event_gen(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # 文件名:FastAPI实例名
        host="0.0.0.0",
        port=8089,
        reload=True  # 热重载，改代码自动重启
    )