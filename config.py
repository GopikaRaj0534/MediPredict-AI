"""
config.py
---------
Centralized configuration management for MediPredict AI.
Handles reliable project-root .env loading, environment variable extraction,
placeholder detection, and safe logging diagnostics.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project Root Directory (unambiguous absolute path)
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Explicitly load .env from project root directory without overwriting existing system env variables
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=False)
else:
    load_dotenv(override=False)


def _clean_key(val):
    """Sanitizes key inputs and filters out empty strings or default placeholders."""
    if not val:
        return None
    val = str(val).strip().strip("'").strip('"')
    if not val or val == "YOUR_GEMINI_API_KEY_HERE" or val.startswith("YOUR_") or val == "your_api_key_here":
        return None
    return val


# Secret Key for Flask Sessions
SECRET_KEY = os.getenv("SECRET_KEY") or "medipredict_default_secret_key_2026"


# Gemini API Helpers
def get_gemini_api_key():
    """Returns the validated Gemini API key string or None if unconfigured/placeholder."""
    for key_name in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_APIKEY", "GEMINI_KEY", "GOOGLE_GEMINI_API_KEY"]:
        cleaned = _clean_key(os.getenv(key_name))
        if cleaned:
            return cleaned
    return None


def get_gemini_model():
    """Returns the configured Gemini model name, stripped of any leading 'models/' prefix for API consistency."""
    model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip()
    if model.startswith("models/"):
        model = model[7:]
    return model or "gemini-flash-lite-latest"


def is_gemini_configured():
    """Returns True if a valid non-placeholder Gemini API key is present."""
    return get_gemini_api_key() is not None


# Google Places API Helpers
def get_places_api_key():
    """Returns the validated Google Places API key string or None."""
    return _clean_key(os.getenv("GOOGLE_PLACES_API_KEY"))


def is_places_configured():
    """Returns True if a valid Google Places API key is present."""
    return get_places_api_key() is not None


def get_diagnostics():
    """Returns safe diagnostic information without exposing secret values."""
    has_env = ENV_PATH.exists()
    has_key = is_gemini_configured()
    model_name = get_gemini_model()
    return {
        "env_path": str(ENV_PATH),
        "env_exists": "YES" if has_env else "NO",
        "key_detected": "YES" if has_key else "NO",
        "model_name": model_name
    }


def log_startup_diagnostics():
    """Prints safe diagnostic logging on startup WITHOUT exposing any API key secrets."""
    diag = get_diagnostics()
    print("\n==================================================")
    print(f"[Gemini] .env path: {diag['env_path']}")
    print(f"[Gemini] .env exists: {diag['env_exists']}")
    print(f"[Gemini] GEMINI_API_KEY detected: {diag['key_detected']}")
    print(f"[Gemini] Configured Model: {diag['model_name']}")
    print(f"[Gemini] Gemini client initialized: {'YES' if diag['key_detected'] == 'YES' else 'NO'}")
    print("==================================================\n")
