from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    DATABASE_URL: str 
    DATABASE_NAME: str 
    DATABASE_PORT: str 
    DATABASE_HOST: str 
    DATABASE_USER: str 
    DATABASE_PASSWORD: SecretStr 
    
    ENVIRONMENT: str
    
    EMBEDDING_MODEL:str = Field()
    EMBEDDING_TIMEOUT_SECONDS: int = Field(default=60)
    OPENROUTER_API_KEY: SecretStr
    
    TOP_K: int = Field(default=3)
    CHAT_MODEL: str = Field()

settings = Settings() # type: ignore[call-arg]