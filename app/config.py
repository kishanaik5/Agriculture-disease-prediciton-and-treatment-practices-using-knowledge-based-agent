from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Kisaan Sampurna CV Service"
    API_V1_STR: str = "/api/v1"
    
    # Database
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None
    DB_SCHEMA: str = "kissan_cv"  # Dedicated schema for this service

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Gemini
    GEMINI_API_KEY: str
    GEMINI_MODEL_FLASH: str = "gemini-2.5-flash"
    GEMINI_MODEL_PRO: str = "gemini-2.5-pro"
    GEMINI_MODEL_BBOX: str = "gemini-3-pro-preview"

    # AWS S3
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_PRIVATE: str
    S3_BUCKET_PUBLIC: str
    
    # Extra
    ENV_MODE: str = "dev"
    SECRET_KEY: str = "CHANGE_THIS"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    GOOGLE_SHEETS_CREDENTIALS_FILE: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore" # Allow extra fields if any other appear

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
