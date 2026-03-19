from typing import Callable, Union, Awaitable, AsyncGenerator, Generator

LLMAdapter = Callable[..., Union[
    str,
    Awaitable[str],
    Generator[str, None, None],
    AsyncGenerator[str, None],
    Awaitable[Generator[str, None, None]],
    Awaitable[AsyncGenerator[str, None]],
]]
