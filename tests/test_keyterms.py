"""Tests for keyterm prompting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import urllib.parse

from deepgram import AsyncDeepgramClient
import pytest

from custom_components.deepgram_stt.config_flow import OptionsFlowHandler
from custom_components.deepgram_stt.const import (
    CONF_KEYTERMS,
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_USE_HA_NAMES,
    MAX_KEYTERMS,
)
from custom_components.deepgram_stt.keyterms import async_build_keyterms, parse_keyterms, supports_keyterms
from custom_components.deepgram_stt.stt import DeepgramSTTEntity
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResultState,
)
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers import area_registry as ar, entity_registry as er

EXPOSE = "custom_components.deepgram_stt.keyterms.async_should_expose"


def test_parse_keyterms() -> None:
    """Commas and newlines both separate terms; blanks are dropped."""
    assert parse_keyterms(None) == []
    assert parse_keyterms("") == []
    assert parse_keyterms("Hue, woonkamer lamp\n\n Sonos ,") == ["Hue", "woonkamer lamp", "Sonos"]


def test_supports_keyterms() -> None:
    """Only Nova-3 models support keyterm prompting."""
    assert supports_keyterms("nova-3")
    assert supports_keyterms("nova-3-general")
    assert not supports_keyterms("nova-2")
    assert not supports_keyterms("whisper-large")


async def test_build_keyterms_manual_only(hass: HomeAssistant) -> None:
    """Without HA names, only manual terms are returned, deduplicated."""
    hass.states.async_set("light.kitchen", "on", {"friendly_name": "Kitchen Light"})
    with patch(EXPOSE, return_value=True):
        assert async_build_keyterms(hass, "Hue, hue, Sonos", use_ha_names=False) == ["Hue", "Sonos"]


async def test_build_keyterms_ha_names(hass: HomeAssistant) -> None:
    """Areas, exposed entity names and aliases are added after manual terms."""
    area = ar.async_get(hass).async_create("Woonkamer", aliases={"Living"})
    entity_registry = er.async_get(hass)
    entry = entity_registry.async_get_or_create("light", "test", "1", suggested_object_id="lamp")
    entity_registry.async_update_entity(entry.entity_id, aliases={"Leeslamp"}, area_id=area.id)
    hass.states.async_set(entry.entity_id, "on", {"friendly_name": "Staande Lamp"})
    hass.states.async_set("switch.hidden", "off", {"friendly_name": "Hidden Switch"})

    with patch(EXPOSE, side_effect=lambda _hass, _assistant, entity_id: entity_id != "switch.hidden"):
        keyterms = async_build_keyterms(hass, "Sonos, woonkamer", use_ha_names=True)

    assert keyterms[:2] == ["Sonos", "woonkamer"]
    assert set(keyterms) == {"Sonos", "woonkamer", "Living", "Staande Lamp", "Leeslamp"}
    assert "Hidden Switch" not in keyterms


async def test_build_keyterms_caps_count_and_length(hass: HomeAssistant) -> None:
    """Overlong terms are skipped and the total is capped."""
    manual = "\n".join(["x" * 60, *(f"term {i}" for i in range(MAX_KEYTERMS + 20))])
    keyterms = async_build_keyterms(hass, manual, use_ha_names=False)
    assert len(keyterms) == MAX_KEYTERMS
    assert keyterms[0] == "term 0"


async def test_options_flow(hass: HomeAssistant) -> None:
    """The options flow shows a form and stores the submitted options."""
    flow = OptionsFlowHandler()
    flow.hass = hass
    entry = MagicMock(options={CONF_KEYTERMS: "Hue"})
    with patch.object(OptionsFlowHandler, "config_entry", entry):
        result = await flow.async_step_init()
        assert result["type"] == "form"
        assert result["step_id"] == "init"

        result = await flow.async_step_init({CONF_KEYTERMS: "Hue, Sonos", CONF_USE_HA_NAMES: False})
        assert result["type"] == "create_entry"
        assert result["data"] == {CONF_KEYTERMS: "Hue, Sonos", CONF_USE_HA_NAMES: False}


def _metadata() -> SpeechMetadata:
    return SpeechMetadata(
        language="nl",
        format=AudioFormats.WAV,
        codec=AudioCodecs.PCM,
        bit_rate=AudioBitRates.BITRATE_16,
        sample_rate=AudioSampleRates.SAMPLERATE_16000,
        channel=AudioChannels.CHANNEL_MONO,
    )


@pytest.mark.parametrize(("model", "expect_keyterms"), [("nova-3", True), ("nova-2", False)])
async def test_end_to_end_with_real_sdk(hass: HomeAssistant, fake_deepgram, model: str, expect_keyterms: bool) -> None:
    """Run the real SDK against a fake socket and check the URL and transcript."""
    hass.states.async_set("light.lamp", "on", {"friendly_name": "Staande Lamp"})
    entry = MagicMock(
        data={CONF_API_KEY: "test_api_key_12345", CONF_MODEL: model, CONF_LANGUAGE: "nl"},
        options={CONF_KEYTERMS: "Sonos", CONF_USE_HA_NAMES: True},
    )
    entity = DeepgramSTTEntity(entry, AsyncDeepgramClient(api_key="test_api_key_12345"))
    entity.hass = hass

    socket = fake_deepgram.script(after_close=[fake_deepgram.result("zet de staande lamp aan", is_final=True)])
    with patch(EXPOSE, return_value=True):
        result = await entity.async_process_audio_stream(_metadata(), fake_deepgram.audio())

    assert result.result == SpeechResultState.SUCCESS
    assert result.text == "zet de staande lamp aan"

    query = urllib.parse.parse_qs(urllib.parse.urlparse(fake_deepgram.urls[0]).query)
    assert query["model"] == [model]
    if expect_keyterms:
        assert query["keyterm"] == ["Sonos", "Staande Lamp"]
    else:
        assert "keyterm" not in query
    assert '{"type": "CloseStream"}' in socket.sent
