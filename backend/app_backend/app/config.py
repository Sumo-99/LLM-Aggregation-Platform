class Settings:
    APP_NAME = "Model Service"
    VERSION = "1.0.0"
    MODEL_API_MAP = {
        "Gemini": "chat_gemini",
        "openai": "chat_openai"
    }

settings = Settings()