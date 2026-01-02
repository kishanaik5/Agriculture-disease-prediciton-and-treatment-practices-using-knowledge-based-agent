from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import init_settings

settings = init_settings()

connect_args = {
    "timeout": 60,
    "server_settings": {
        "application_name": settings.PROJECT_NAME
    }
}

# Only require SSL in production or non-dev environments, OR if connecting to RDS
if settings.ENV_MODE != "dev" or "rds.amazonaws.com" in settings.POSTGRES_SERVER:
    connect_args["ssl"] = "require"

engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    echo=True,
    pool_pre_ping=True,
    connect_args=connect_args
)

SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)



async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
