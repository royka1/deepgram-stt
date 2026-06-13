"""Deepgram Speech-to-Text platform."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
import deepgram.extensions.types.sockets as deepgram_socket_types
from deepgram.extensions.types.sockets import ListenV1ControlMessage
from deepgram.listen.v1 import client as deepgram_listen_client, raw_client as deepgram_listen_raw_client

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util.ssl import client_context

from .const import (
    CONF_API_KEY,
    CONF_LANGUAGE,
    CONF_MODEL,
    DEFAULT_ENCODING,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_SAMPLE_RATE,
    KEEPALIVE_INTERVAL,
    STREAM_DELAY,
    TRANSCRIPT_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)


def _wrap_connect_with_ssl(original_connect: Any, ssl_context: Any) -> Any:
    """Wrap a websockets connect callable to default to a pre-built SSL context."""

    def _connect_with_preloaded_ssl(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("ssl", ssl_context)
        return original_connect(*args, **kwargs)

    _connect_with_preloaded_ssl._ha_ssl_patched = True  # noqa: SLF001
    return _connect_with_preloaded_ssl


def _prepare_client(api_key: str) -> AsyncDeepgramClient:
    """Create the Deepgram client, doing all blocking work off the event loop.

    The SDK loads SSL certificates when the client is created and again when a
    websocket connects, and lazily imports its message-type modules on first
    use. All of that blocks, so this function must run in an executor.
    """
    # Materialize the package's lazy attributes so its module __getattr__
    # (which calls import_module) never runs in the event loop
    for name in deepgram_socket_types.__all__:
        setattr(deepgram_socket_types, name, getattr(deepgram_socket_types, name))

    # The SDK offers no way to pass an SSL context to its websocket connect, so
    # it would build a new one (loading CA certs) inside the event loop on every
    # connection. Both the listen client and its raw client hold their own
    # binding of websockets' connect; wrap each to inject Home Assistant's
    # cached context. AsyncV1Client.connect uses the binding in client.py.
    ssl_context = client_context()
    for ws_module in (deepgram_listen_client, deepgram_listen_raw_client):
        if not getattr(ws_module.websockets_client_connect, "_ha_ssl_patched", False):
            ws_module.websockets_client_connect = _wrap_connect_with_ssl(
                ws_module.websockets_client_connect, ssl_context
            )

    client = AsyncDeepgramClient(api_key=api_key)
    # First access lazily imports the listen client modules and caches the
    # result on the instance; trigger that here instead of at connect time
    _ = client.listen.v1
    return client


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Deepgram STT from config entry."""
    api_key = config_entry.data.get(CONF_API_KEY)
    client = await hass.async_add_executor_job(_prepare_client, api_key) if api_key else None
    async_add_entities([DeepgramSTTEntity(config_entry, client)])


class DeepgramSTTEntity(SpeechToTextEntity):
    """Deepgram Speech-to-Text entity."""

    def __init__(self, config_entry: ConfigEntry, client: AsyncDeepgramClient | None = None) -> None:
        """Initialize Deepgram STT."""
        self._attr_name = "Deepgram STT"
        self._attr_unique_id = "deepgram_stt"
        self._client = client
        self._api_key = config_entry.data.get(CONF_API_KEY)
        self._model = config_entry.data.get(CONF_MODEL, DEFAULT_MODEL)
        self._language = config_entry.data.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        self.config_entry = config_entry
        self._attr_device_info = None  # No physical device for cloud API

    @property
    def api_key(self) -> str:
        """Return configured API key."""
        return self._api_key

    @property
    def model(self) -> str:
        """Return configured model."""
        return self._model

    @property
    def language(self) -> str:
        """Return configured language."""
        return self._language

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return ["pl", "en", "de", "es", "fr", "it", "nl", "pt"]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return supported audio formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return supported audio codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return supported bit rates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return supported sample rates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return supported audio channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(self, metadata: SpeechMetadata, stream: asyncio.StreamReader) -> SpeechResult:
        """Process audio stream with Deepgram."""
        if not self._api_key:
            _LOGGER.error("Deepgram API key not configured")
            return SpeechResult("", SpeechResultState.ERROR)

        try:
            # The client is normally created in an executor during setup;
            # creating it here would block the event loop loading SSL certs.
            client = self._client if self._client is not None else AsyncDeepgramClient(api_key=self._api_key)

            # Storage for transcript
            transcript_parts = []
            final_transcript = ""
            is_final = False
            error_occurred = False
            state_lock = asyncio.Lock()

            # Event handlers
            async def on_message(message, **kwargs):
                nonlocal transcript_parts, final_transcript, is_final

                _LOGGER.debug("Received message from Deepgram: type=%s", type(message).__name__)

                if not hasattr(message, "channel"):
                    # Non-transcript events such as Metadata or SpeechStarted
                    _LOGGER.debug("Ignoring %s event from Deepgram", getattr(message, "type", type(message).__name__))
                    return

                if not message.channel.alternatives:
                    _LOGGER.warning("Message channel has no alternatives")
                    return

                sentence = message.channel.alternatives[0].transcript

                if len(sentence) > 0:
                    async with state_lock:
                        if message.is_final:
                            final_transcript = sentence
                            is_final = True
                            _LOGGER.debug("Final transcript: %s", sentence)
                        else:
                            transcript_parts.append(sentence)
                            _LOGGER.debug("Interim transcript: %s", sentence)
                else:
                    _LOGGER.debug("Received empty transcript (silence detection)")

            async def on_error(error, **kwargs):
                nonlocal error_occurred
                _LOGGER.error("Deepgram error: %s", error)
                error_occurred = True

            # Connect to Deepgram with async context manager
            # Use v1 API - v2 doesn't support language parameter in Python SDK
            async with client.listen.v1.connect(
                model=self._model,
                language=self._language,
                encoding=DEFAULT_ENCODING,
                sample_rate=DEFAULT_SAMPLE_RATE,
                channels=1,
                interim_results=True,
            ) as dg_connection:
                # Register event handlers
                dg_connection.on(EventType.MESSAGE, on_message)
                dg_connection.on(EventType.ERROR, on_error)

                # Start listening for messages in background
                listen_task = asyncio.create_task(dg_connection.start_listening())
                _LOGGER.debug("Deepgram connection started, listening task created")

                # Keep the socket alive through audio pauses, e.g. while the
                # pipeline is still waiting for speech (Deepgram NET-0001)
                async def send_keepalives() -> None:
                    while True:
                        await asyncio.sleep(KEEPALIVE_INTERVAL)
                        try:
                            await dg_connection.send_control(ListenV1ControlMessage(type="KeepAlive"))
                        except Exception:  # noqa: BLE001 - connection already closing
                            return

                keepalive_task = asyncio.create_task(send_keepalives())

                # Stream audio data
                try:
                    chunk_count = 0
                    total_bytes = 0
                    async for chunk in stream:
                        chunk_size = len(chunk)
                        total_bytes += chunk_size
                        chunk_count += 1
                        await dg_connection.send_media(chunk)
                        await asyncio.sleep(STREAM_DELAY)

                    _LOGGER.debug("Audio streaming complete: %d chunks, %d bytes total", chunk_count, total_bytes)

                    # Send close stream signal to finalize transcription
                    await dg_connection.send_control(ListenV1ControlMessage(type="CloseStream"))
                    _LOGGER.debug("Sent CloseStream signal to Deepgram")

                    # Wait for final transcript (with timeout)
                    start_time = asyncio.get_event_loop().time()
                    while True:
                        async with state_lock:
                            if is_final or error_occurred:
                                break
                        if asyncio.get_event_loop().time() - start_time > TRANSCRIPT_TIMEOUT:
                            _LOGGER.warning("Timeout waiting for final transcript")
                            break
                        await asyncio.sleep(0.1)

                except Exception as e:  # noqa: BLE001
                    _LOGGER.error("Error streaming audio: %s", e)
                    return SpeechResult("", SpeechResultState.ERROR)
                finally:
                    # Cancel background tasks if still running
                    for task in (keepalive_task, listen_task):
                        if not task.done():
                            task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await task

            # Return result (connection auto-closed by context manager)
            async with state_lock:
                if error_occurred:
                    return SpeechResult("", SpeechResultState.ERROR)

                result_text = final_transcript if final_transcript else " ".join(transcript_parts)

            if not result_text:
                _LOGGER.warning("No transcript received")
                return SpeechResult("", SpeechResultState.ERROR)

            _LOGGER.info("Transcription result: %s", result_text)
            return SpeechResult(result_text, SpeechResultState.SUCCESS)

        except Exception as e:  # noqa: BLE001
            _LOGGER.error("Deepgram transcription failed: %s", e)
            return SpeechResult("", SpeechResultState.ERROR)
