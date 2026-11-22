"""Config flow for Deepgram STT integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import BooleanSelector, TextSelector, TextSelectorConfig

from .const import (
    CONF_API_KEY,
    CONF_KEYTERMS,
    CONF_LANGUAGE,
    CONF_MODEL,
    CONF_USE_HA_NAMES,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_USE_HA_NAMES,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): str,
        vol.Optional(CONF_LANGUAGE, default=DEFAULT_LANGUAGE): str,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    # Basic validation - ensure API key is not empty
    data[CONF_API_KEY] = data[CONF_API_KEY].strip()
    if not data[CONF_API_KEY] or len(data[CONF_API_KEY]) < 10:
        raise InvalidAPIKey

    # Return info to be stored in config entry
    return {"title": "Deepgram STT"}


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Deepgram STT."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry) -> OptionsFlowHandler:
        """Return the options flow handler."""
        return OptionsFlowHandler()

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            try:
                info = await validate_input(self.hass, user_input)
            except InvalidAPIKey:
                errors["base"] = "invalid_api_key"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                # Prevent multiple instances
                await self.async_set_unique_id(DOMAIN)
                self._abort_if_unique_id_configured()

                return self.async_create_entry(title=info["title"], data=user_input)

        return self.async_show_form(step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors)

    async def async_step_import(self, import_data: dict[str, Any]) -> FlowResult:
        """Handle import from async_setup (auto-configuration from sops secret)."""
        # Prevent multiple instances
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        return self.async_create_entry(title="Deepgram STT", data=import_data)


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle Deepgram STT options (keyterm prompting)."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)

        options = self.config_entry.options
        schema = vol.Schema(
            {
                vol.Optional(
                    CONF_KEYTERMS,
                    description={"suggested_value": options.get(CONF_KEYTERMS, "")},
                ): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Optional(
                    CONF_USE_HA_NAMES,
                    default=options.get(CONF_USE_HA_NAMES, DEFAULT_USE_HA_NAMES),
                ): BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)


class InvalidAPIKey(HomeAssistantError):
    """Error to indicate API key is invalid."""
