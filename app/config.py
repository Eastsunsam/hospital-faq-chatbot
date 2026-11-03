"""
Application Configuration
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Explicitly load .env from workspace root
load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_TITLE: str = "Hospital FAQ Chatbot API"
    APP_DESCRIPTION: str = "Conversational hospital assistant powered by OpenAI and structured hospital knowledge base."
    APP_VERSION: str = "1.0.0"

    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 600

    # Knowledge Base file path
    KNOWLEDGE_BASE_PATH: Path = BASE_DIR / "hospital_service.json"
    STATIC_DIR: Path = BASE_DIR / "static"

    @property
    def is_api_key_configured(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY != "your_openai_api_key_here")


settings = Settings()
