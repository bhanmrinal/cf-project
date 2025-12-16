"""Tests for configuration module."""

from backend.app.core.config import Settings, LLMProvider


def test_settings_defaults():
    """Test that default settings are loaded correctly."""
    settings = Settings()

    assert settings.app_name == "Careerflow Resume Optimizer"
    assert settings.llm_provider == LLMProvider.GROQ
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.port == 8000


def test_llm_provider_enum():
    """Test LLM provider enumeration."""
    assert LLMProvider.GROQ.value == "groq"
    assert LLMProvider.HUGGINGFACE.value == "huggingface"
    assert LLMProvider.OLLAMA.value == "ollama"


def test_current_llm_model():
    """Test current LLM model property."""
    settings = Settings()

    settings.llm_provider = LLMProvider.GROQ
    assert settings.current_llm_model == settings.groq_model

    settings.llm_provider = LLMProvider.OLLAMA
    assert settings.current_llm_model == settings.ollama_model


def test_firebase_credentials_none_when_incomplete():
    """Test that Firebase credentials return None when incomplete."""
    settings = Settings()

    # With no Firebase config, should return None
    assert settings.firebase_credentials is None
