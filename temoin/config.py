"""Application settings loaded from environment variables or Streamlit secrets.

Every tunable constant of the project lives here and nowhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when the application is misconfigured."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings with safe defaults."""

    baseten_api_key: str
    model_name: str = "meta-models/muse-spark-open-1026"
    model_base_url: str = "https://inference.baseten.co/v1"
    model_timeout_seconds: float = 120.0
    model_max_attempts: int = 2
    model_retry_wait_seconds: float = 2.0
    model_json_retries: int = 1
    image_max_pixels: int = 1024
    image_max_megabytes: float = 15.0
    image_jpeg_quality: int = 90
    cache_dir: Path = Path(".cache/temoin")
    agent_max_steps: int = 8
    human_review_disclaimer: str = "Témoin assiste une décision humaine, il ne la remplace pas."
    challenge_alphabet: str = "ABCDEFGHJKLMNPQRTUVWXYZ2346789"
    challenge_code_length: int = 2
    challenge_validity_minutes: int = 15
    challenge_max_issues: int = 100
    data_dir: Path = Path("data")
    model_api_url: str = "https://api.meta.ai/v1"
    model_api_key: str = ""
    video_max_frames: int = 6
    video_max_megabytes: float = 200.0
    forensics_ela_quality: int = 90
    forensics_noise_ratio: float = 0.12
    forensics_blur_min: float = 8.0
    stereo_min_matches: int = 30
    stereo_orb_features: int = 2000
    stereo_num_disparities: int = 64
    stereo_block_size: int = 7
    stereo_uniqueness: int = 10
    stereo_max_width: int = 240


def _read_streamlit_secrets() -> dict[str, str]:
    """Return Streamlit secrets, or an empty dict when unavailable."""
    try:
        import streamlit as st

        return {str(key): str(value) for key, value in dict(st.secrets).items()}
    except Exception:
        return {}


def load_settings(env_file: str | None = None) -> Settings:
    """Load settings from Streamlit secrets first, then environment and .env."""
    load_dotenv(env_file)
    secrets = _read_streamlit_secrets()

    def get(name: str, default: str = "") -> str:
        """Read one setting, preferring Streamlit secrets over environment."""
        return str(secrets.get(name, os.environ.get(name, default))).strip()

    api_key = get("BASETEN_API_KEY") or get("MODEL_API_KEY")
    if not api_key:
        raise ConfigError(
            "La clé d'accès au modèle est manquante. Ajoutez BASETEN_API_KEY "
            "dans le fichier .env à la racine du projet, puis relancez."
        )
    return Settings(
        baseten_api_key=api_key,
        model_name=get("MODEL_NAME") or Settings.model_name,
        model_base_url=get("MODEL_BASE_URL") or Settings.model_base_url,
        model_api_key=get("MODEL_API_KEY"),
        model_api_url=get("MODEL_API_URL") or Settings.model_api_url,
    )
