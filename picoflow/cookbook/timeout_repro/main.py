import argparse
import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict

from picoflow import create_agent, llm


def make_handler(response_delay: float, stream_chunk_delay: float):
    class MockOpenAIHandler(BaseHTTPRequestHandler):
        server_version = "MockOpenAI/1.0"
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404, "Not Found")
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return

            raw = self.rfile.read(content_length)
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self.send_error(400, "Invalid JSON")
                return

            stream = bool(payload.get("stream"))
            prompt = ""
            messages = payload.get("messages") or []
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    prompt = str(last.get("content") or "")

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                chunks = [
                    f"Echo: {prompt[:12]}",
                    " | slow",
                    " | stream",
                ]
                try:
                    for piece in chunks:
                        data = {
                            "choices": [
                                {
                                    "delta": {"content": piece},
                                }
                            ]
                        }
                        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                        time.sleep(stream_chunk_delay)

                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            time.sleep(response_delay)
            body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Echo: {prompt}",
                        }
                    }
                ]
            }
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            try:
                self.wfile.write(encoded)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, fmt: str, *args):
            return

    return MockOpenAIHandler


def start_server(response_delay: float, stream_chunk_delay: float) -> Dict[str, object]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    sock.close()

    handler = make_handler(response_delay, stream_chunk_delay)
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return {"server": server, "thread": thread, "host": host, "port": port}


def build_agent(dsn: str):
    return create_agent(llm("{input}", llm_adapter=dsn))


async def run_non_stream(agent, prompt: str, timeout: float):
    return await agent.arun(prompt, timeout=timeout)


async def run_stream(agent, prompt: str, timeout: float):
    chunks = []

    async def on_chunk(text: str):
        chunks.append(text)

    state = await agent.arun(prompt, timeout=timeout, stream_callback=on_chunk)
    return state, "".join(chunks)


def main():
    parser = argparse.ArgumentParser(description="Reproduce PicoFlow timeout behavior locally.")
    parser.add_argument(
        "--scenario",
        choices=["agent-timeout", "request-timeout", "stream-request-timeout"],
        default="agent-timeout",
    )
    parser.add_argument("--agent-timeout", type=float, default=2.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--response-delay", type=float, default=4.0)
    parser.add_argument("--stream-chunk-delay", type=float, default=3.0)
    args = parser.parse_args()

    server_info = start_server(
        response_delay=args.response_delay,
        stream_chunk_delay=args.stream_chunk_delay,
    )
    server = server_info["server"]
    host = server_info["host"]
    port = server_info["port"]

    if args.scenario == "agent-timeout":
        dsn_timeout = max(args.request_timeout, args.response_delay + 1)
        agent_timeout = args.agent_timeout
        use_stream = False
    elif args.scenario == "request-timeout":
        dsn_timeout = min(args.request_timeout, max(args.response_delay - 1, 0.1))
        agent_timeout = max(args.agent_timeout, args.response_delay + 2)
        use_stream = False
    else:
        dsn_timeout = min(args.request_timeout, max(args.stream_chunk_delay - 1, 0.1))
        agent_timeout = max(args.agent_timeout, args.stream_chunk_delay * 4)
        use_stream = True

    dsn = f"llm+openai://{host}:{port}/mock-model?api_key=none&timeout={dsn_timeout}"
    agent = build_agent(dsn)

    print(f"Scenario: {args.scenario}")
    print(f"DSN: {dsn}")
    print(f"Agent timeout: {agent_timeout}")
    print(f"Response delay: {args.response_delay}")
    print(f"Stream chunk delay: {args.stream_chunk_delay}")

    try:
        if use_stream:
            state, streamed = asyncio.run(
                run_stream(agent, "stream timeout demo", timeout=agent_timeout)
            )
            print(f"Final output: {state.output}")
            print(f"Streamed chunks: {streamed}")
        else:
            state = asyncio.run(
                run_non_stream(agent, "timeout demo", timeout=agent_timeout)
            )
            print(f"Final output: {state.output}")
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
