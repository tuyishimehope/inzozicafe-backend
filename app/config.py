from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ENVIRONMENT: str

    DATABASE_HOST: str
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: SecretStr

    OPENROUTER_API_KEY: SecretStr
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL: str
    CHAT_MODEL: str
    LLM_TIMEOUT_SECONDS: int = 60

    TOP_K: int = 3

    @property
    def database_url(self) -> str:
        password = self.DATABASE_PASSWORD.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.DATABASE_USER}:{password}"
            f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )


settings = Settings()  # type: ignore[call-arg]
