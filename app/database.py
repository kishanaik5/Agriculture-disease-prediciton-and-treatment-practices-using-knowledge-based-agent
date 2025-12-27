from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    echo=True,
    pool_pre_ping=True,
    connect_args={
        "ssl": "require",
        "timeout": 60,
        "server_settings": {
            "application_name": settings.PROJECT_NAME
        }
    }
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
