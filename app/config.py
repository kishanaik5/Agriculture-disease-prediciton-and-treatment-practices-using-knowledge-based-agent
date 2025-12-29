from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator, ValidationError
from functools import lru_cache
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application Settings
    
    Required Environment Variables:
    - POSTGRES_SERVER, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
    - GEMINI_API_KEY
    - AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    - S3_BUCKET_PRIVATE, S3_BUCKET_PUBLIC
    """
    
    PROJECT_NAME: str = "Kisaan Sampurna CV Service"
    API_V1_STR: str = "/api/v1"
    
    # Database Configuration
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int = 5432
    DATABASE_URL: Optional[str] = None
    DB_SCHEMA: str = "kissan_cv"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Build database URI from components or use DATABASE_URL directly"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Gemini AI Configuration
    GEMINI_API_KEY: str
    GEMINI_MODEL_FLASH: str = "gemini-2.0-flash-exp"
    GEMINI_MODEL_PRO: str = "gemini-1.5-pro"
    GEMINI_MODEL_BBOX: str = "gemini-2.0-flash-exp"  # Alternative: try gemini-1.5-pro if bbox fails

    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "ap-south-1"
    S3_BUCKET_PRIVATE: str
    S3_BUCKET_PUBLIC: str
    
    # Redis Configuration for Async Tasks
    REDIS_URL: str = "redis://localhost:6379"
    
    # Application Configuration
    ENV_MODE: str = "dev"

    @field_validator('GEMINI_API_KEY')
    @classmethod
    def validate_credentials(cls, v: str) -> str:
        """Ensure credentials are not empty"""
        if not v or v.strip() == '':
            raise ValueError("Credential cannot be empty")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"
        case_sensitive = True


def get_settings() -> Settings:
    """
    Get application settings with detailed error reporting.
    
    Raises:
        SystemExit: If required environment variables are missing
    """
    try:
        settings = Settings()
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"📊 Environment: {settings.ENV_MODE}")
        logger.info(f"🗄️  Database: {settings.POSTGRES_SERVER}/{settings.POSTGRES_DB}")
        logger.info(f"🔧 Schema: {settings.DB_SCHEMA}")
        logger.info(f"☁️  S3 Region: {settings.AWS_REGION}")
        return settings
    except ValidationError as e:
        logger.error("❌ Configuration validation failed!")
        logger.error("=" * 60)
        logger.error("MISSING OR INVALID ENVIRONMENT VARIABLES:")
        logger.error("=" * 60)
        
        for error in e.errors():
            field = error['loc'][0]
            error_type = error['type']
            msg = error['msg']
            
            logger.error(f"  ❌ {field}")
            logger.error(f"     Type: {error_type}")
            logger.error(f"     Message: {msg}")
            logger.error("")
        
        logger.error("=" * 60)
        logger.error("REQUIRED ENVIRONMENT VARIABLES:")
        logger.error("=" * 60)
        logger.error("Database:")
        logger.error("  - POSTGRES_SERVER")
        logger.error("  - POSTGRES_USER")
        logger.error("  - POSTGRES_PASSWORD")
        logger.error("  - POSTGRES_DB")
        logger.error("")
        logger.error("AI/ML:")
        logger.error("  - GEMINI_API_KEY")
        logger.error("")
        logger.error("AWS:")
        logger.error("  - AWS_ACCESS_KEY_ID")
        logger.error("  - AWS_SECRET_ACCESS_KEY")
        logger.error("  - S3_BUCKET_PRIVATE")
        logger.error("  - S3_BUCKET_PUBLIC")
        logger.error("=" * 60)
        logger.error("Please set these variables in your .env file or environment")
        logger.error("See .env.example for reference")
        logger.error("=" * 60)
        
        # In production/staging, exit gracefully
        # In dev, you might want to continue with partial config
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error loading configuration: {e}")
        sys.exit(1)


# Singleton settings instance
settings: Optional[Settings] = None

def init_settings() -> Settings:
    """Initialize settings (called once at startup)"""
    global settings
    if settings is None:
        settings = get_settings()
    return settings
