"""Tests for settings loading (no secret is ever displayed)."""

from __future__ import annotations

import pytest

from temoin.config import ConfigError, Settings, load_settings


@pytest.fixture(autouse=True)
def _no_streamlit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from any real Streamlit secrets file."""
    monkeypatch.setattr("temoin.config._read_streamlit_secrets", lambda: {})


def test_load_settings_reads_api_key(monkeypatch: pytest.MonkeyPatch, empty_env_file: str) -> None:
    """The Baseten key is loaded and defaults are applied."""
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "key-123")
    settings = load_settings(env_file=empty_env_file)
    assert isinstance(settings, Settings)
    assert settings.baseten_api_key == "key-123"
    assert settings.model_name == "meta-models/muse-spark-open-1026"
    assert settings.model_base_url == "https://inference.baseten.co/v1"


def test_load_settings_falls_back_to_model_api_key(
    monkeypatch: pytest.MonkeyPatch, empty_env_file: str
) -> None:
    """MODEL_API_KEY is accepted when BASETEN_API_KEY is missing."""
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "fallback-456")
    settings = load_settings(env_file=empty_env_file)
    assert settings.baseten_api_key == "fallback-456"


def test_load_settings_missing_key_is_plain_language(
    monkeypatch: pytest.MonkeyPatch, empty_env_file: str
) -> None:
    """A missing key raises a clear error naming the variable, not the value."""
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ConfigError) as caught:
        load_settings(env_file=empty_env_file)
    assert "BASETEN_API_KEY" in str(caught.value)
    assert ".env" in str(caught.value)


def test_load_settings_reads_meta_api_fields(
    monkeypatch: pytest.MonkeyPatch, empty_env_file: str
) -> None:
    """MODEL_API_KEY and MODEL_API_URL load; the URL has a documented default."""
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_URL", raising=False)
    monkeypatch.setenv("BASETEN_API_KEY", "key-123")
    assert load_settings(env_file=empty_env_file).model_api_key == ""
    assert load_settings(env_file=empty_env_file).model_api_url == "https://api.meta.ai/v1"
    monkeypatch.setenv("MODEL_API_KEY", "meta-789")
    monkeypatch.setenv("MODEL_API_URL", "https://meta.example/v9")
    settings = load_settings(env_file=empty_env_file)
    assert (settings.model_api_key, settings.model_api_url) == (
        "meta-789",
        "https://meta.example/v9",
    )
