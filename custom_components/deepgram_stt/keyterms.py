"""Build the keyterm list sent to Deepgram for keyterm prompting."""

from __future__ import annotations

from collections.abc import Iterable
import re

from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import area_registry as ar, entity_registry as er

from .const import MAX_KEYTERM_LENGTH, MAX_KEYTERMS

_ASSISTANT = "conversation"


def parse_keyterms(raw: str | None) -> list[str]:
    """Split user-entered keyterms on commas and newlines."""
    if not raw:
        return []
    return [term.strip() for term in re.split(r"[,\n]", raw) if term.strip()]


def supports_keyterms(model: str) -> bool:
    """Return whether the model supports keyterm prompting (Nova-3 only)."""
    return model.startswith("nova-3")


@callback
def _ha_names(hass: HomeAssistant) -> Iterable[str]:
    """Yield area names and names of entities exposed to Assist, with aliases."""
    for area in ar.async_get(hass).async_list_areas():
        yield area.name
        yield from area.aliases

    entity_registry = er.async_get(hass)
    for state in hass.states.async_all():
        if not async_should_expose(hass, _ASSISTANT, state.entity_id):
            continue
        yield state.name
        if (entry := entity_registry.async_get(state.entity_id)) is not None:
            yield from entry.aliases


@callback
def async_build_keyterms(hass: HomeAssistant, manual: str | None, use_ha_names: bool) -> list[str]:
    """Return deduplicated keyterms: manual terms first, then Home Assistant names."""
    candidates: list[Iterable[str]] = [parse_keyterms(manual)]
    if use_ha_names:
        candidates.append(_ha_names(hass))

    keyterms: list[str] = []
    seen: set[str] = set()
    for source in candidates:
        for raw_term in source:
            term = raw_term.strip() if raw_term else ""
            key = term.casefold()
            if not term or len(term) > MAX_KEYTERM_LENGTH or key in seen:
                continue
            seen.add(key)
            keyterms.append(term)
            if len(keyterms) >= MAX_KEYTERMS:
                return keyterms
    return keyterms
