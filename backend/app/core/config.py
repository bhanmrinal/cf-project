"""
Application configuration using Pydantic Settings.

Supports multiple LLM providers:
- Groq (Llama 3.3, Mixtral) - Recommended, fast and free
- HuggingFace Inference API
- Local models via Ollama
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Careerflow Resume Optimizer"
    app_env: str = "development"
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    secret_key: str = "your-secret-key-change-in-production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM Provider Selection
    llm_provider: LLMProvider = LLMProvider.GROQ

    # Groq Configuration (Llama 3.3 70B, Mixtral 8x7B)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"  # Options: llama-3.3-70b-versatile, mixtral-8x7b-32768

    # HuggingFace Configuration
    huggingface_api_key: str = ""
    huggingface_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Ollama Configuration (Local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Embedding Model (using HuggingFace for embeddings)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Firebase
    firebase_project_id: str | None = None
    firebase_private_key_id: str | None = None
    firebase_private_key: str | None = None
    firebase_client_email: str | None = None
    firebase_client_id: str | None = None
    firebase_auth_uri: str = "https://accounts.google.com/o/oauth2/auth"
    firebase_token_uri: str = "https://oauth2.googleapis.com/token"
    firebase_auth_provider_cert_url: str = "https://www.googleapis.com/oauth2/v1/certs"
    firebase_client_cert_url: str | None = None

    # ChromaDB
    chroma_persist_directory: str = "./chroma_db"

    # File Upload
    upload_directory: str = "./uploads"
    max_file_size_mb: int = 10
    allowed_extensions: list[str] = Field(default=["pdf", "docx"])

    class Config:
        env_file = (".env", "../.env")  # Look in current dir and parent dir
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def firebase_credentials(self) -> dict | None:
        """Generate Firebase credentials dictionary from environment variables."""
        if not all(
            [
                self.firebase_project_id,
                self.firebase_private_key,
                self.firebase_client_email,
            ]
        ):
            return None

        private_key = self.firebase_private_key
        if private_key and "\\n" in private_key:
            private_key = private_key.replace("\\n", "\n")

        return {
            "type": "service_account",
            "project_id": self.firebase_project_id,
            "private_key_id": self.firebase_private_key_id,
            "private_key": private_key,
            "client_email": self.firebase_client_email,
            "client_id": self.firebase_client_id,
            "auth_uri": self.firebase_auth_uri,
            "token_uri": self.firebase_token_uri,
            "auth_provider_x509_cert_url": self.firebase_auth_provider_cert_url,
            "client_x509_cert_url": self.firebase_client_cert_url,
        }

    @property
    def current_llm_model(self) -> str:
        """Get the current LLM model name based on provider."""
        provider_models = {
            LLMProvider.GROQ: self.groq_model,
            LLMProvider.HUGGINGFACE: self.huggingface_model,
            LLMProvider.OLLAMA: self.ollama_model,
        }
        return provider_models.get(self.llm_provider, self.groq_model)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
