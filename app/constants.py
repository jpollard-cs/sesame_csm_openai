import os

# Determine a writable base directory.
# If CSM_BASE_DIR env var is set, use it; otherwise use a folder named `csm_data` inside the repo root.
DEFAULT_BASE_DIR = os.getenv(
    "CSM_BASE_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "csm_data")),
)
BASE_DIR = DEFAULT_BASE_DIR

# Ensure the base directory exists
os.makedirs(BASE_DIR, exist_ok=True)

# Convenience helpers for sub‑paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
TOKENIZERS_DIR = os.path.join(BASE_DIR, "tokenizers")
VOICE_MEMORIES_DIR = os.path.join(BASE_DIR, "voice_memories")
VOICE_REFERENCES_DIR = os.path.join(BASE_DIR, "voice_references")
VOICE_PROFILES_DIR = os.path.join(BASE_DIR, "voice_profiles")
CLONED_VOICES_DIR = os.path.join(BASE_DIR, "cloned_voices")
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "audio_cache")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Create all directories if they don't exist
for _d in (
    MODELS_DIR,
    TOKENIZERS_DIR,
    VOICE_MEMORIES_DIR,
    VOICE_REFERENCES_DIR,
    VOICE_PROFILES_DIR,
    CLONED_VOICES_DIR,
    AUDIO_CACHE_DIR,
    STATIC_DIR,
):
    os.makedirs(_d, exist_ok=True)

# VOICE_BACKUPS_DIR for backups of voice data
VOICE_BACKUPS_DIR = os.path.join(BASE_DIR, "voice_backups")
# Ensure backup directory exists
os.makedirs(VOICE_BACKUPS_DIR, exist_ok=True) 