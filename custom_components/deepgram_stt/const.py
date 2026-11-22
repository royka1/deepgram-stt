"""Constants for Deepgram STT integration."""

DOMAIN = "deepgram_stt"

# API Configuration
DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "pl"
DEFAULT_ENCODING = "linear16"
DEFAULT_SAMPLE_RATE = 16000

# Keyterm prompting (nova-3 only). Deepgram caps the total keyterm size per
# request, so limit how many terms are sent.
MAX_KEYTERMS = 100
MAX_KEYTERM_LENGTH = 50
DEFAULT_USE_HA_NAMES = True

# Streaming Configuration
STREAM_DELAY = 0.01
TRANSCRIPT_TIMEOUT = 5
# Deepgram closes the socket (NET-0001) after ~10s without audio or a message
KEEPALIVE_INTERVAL = 5

# Config Keys
CONF_API_KEY = "api_key"
CONF_MODEL = "model"
CONF_LANGUAGE = "language"
CONF_KEYTERMS = "keyterms"
CONF_USE_HA_NAMES = "use_ha_names"
