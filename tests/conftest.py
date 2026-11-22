"""Shared fixtures, including a scripted fake of Deepgram's live websocket."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import contextlib
from dataclasses import dataclass, field
import json
from typing import Any

from deepgram.listen.v1 import client as deepgram_listen_client
import pytest


def results_message(transcript: str, *, is_final: bool) -> str:
    """Return a Deepgram Results message as JSON."""
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": 1.0,
            "start": 0.0,
            "is_final": is_final,
            "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9, "words": []}]},
            "metadata": {
                "request_id": "r",
                "model_info": {"name": "n", "version": "v", "arch": "a"},
                "model_uuid": "u",
            },
        }
    )


class FakeWebSocket:
    """Websocket that sends scripted messages, then closes after CloseStream.

    `before_close` messages are sent right away, as if Deepgram finalized them
    while audio was still streaming; `after_close` messages are sent once the
    client sends CloseStream, after which the socket closes.
    """

    def __init__(self, before_close: list[str] | None = None, after_close: list[str] | None = None) -> None:
        self.sent: list[Any] = []
        self._before = list(before_close or [])
        self._after = list(after_close or [])
        self._closed = asyncio.Event()

    async def send(self, data: Any) -> None:
        self.sent.append(data)
        if isinstance(data, str) and json.loads(data).get("type") == "CloseStream":
            self._closed.set()

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if self._before:
            return self._before.pop(0)
        await self._closed.wait()
        if self._after:
            return self._after.pop(0)
        raise StopAsyncIteration


@dataclass
class FakeDeepgram:
    """Handle returned by the `fake_deepgram` fixture."""

    socket: FakeWebSocket = field(default_factory=FakeWebSocket)
    urls: list[str] = field(default_factory=list)

    def script(self, before_close: list[str] | None = None, after_close: list[str] | None = None) -> FakeWebSocket:
        """Set the messages the next connection receives."""
        self.socket = FakeWebSocket(before_close, after_close)
        return self.socket

    result = staticmethod(results_message)

    @staticmethod
    async def audio() -> AsyncIterator[bytes]:
        """Yield a single chunk of silent 16-bit audio."""
        yield b"\x00\x00" * 160


@pytest.fixture
def fake_deepgram(monkeypatch: pytest.MonkeyPatch) -> FakeDeepgram:
    """Route the real SDK's websocket connect to a scripted fake socket."""
    fake = FakeDeepgram()

    @contextlib.asynccontextmanager
    async def fake_connect(url: str, **_kwargs: Any) -> AsyncIterator[FakeWebSocket]:
        fake.urls.append(url)
        yield fake.socket

    monkeypatch.setattr(deepgram_listen_client, "websockets_client_connect", fake_connect)
    return fake
