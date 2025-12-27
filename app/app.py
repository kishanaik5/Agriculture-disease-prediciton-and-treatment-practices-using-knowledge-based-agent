from fastapi import FastAPI
from app.config import settings
from app.routers.v1.scan import router as scan_router
from SharedBackend.managers.base import BaseSchema
from app.database import engine
from sqlalchemy import text
import asyncio

db_ready = False

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

async def wait_for_db(engine, max_retries: int = 10):
    global db_ready
    delay = 2  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            # Skip table creation - tables should exist via migrations
            # async with engine.begin() as conn:
            #     await conn.run_sync(BaseSchema.metadata.create_all)
            
            # Just verify connection
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                
            db_ready = True
            print("✅ Database connected successfully")
            return
        except Exception as e:
            print(f"⏳ DB not ready (attempt {attempt}/{max_retries}): {e}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)  # exponential backoff

    print("❌ Database failed to connect after retries")

# Verify DB connection on startup (tables should be created via Alembic migrations)
@app.on_event("startup")
async def startup():
    asyncio.create_task(wait_for_db(engine))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.include_router(scan_router, prefix=settings.API_V1_STR, tags=["scan"])

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/health/db")
async def db_health():
    if db_ready:
        return {"db": "ok"}
    
    # Optional: try one active check if flag is false
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return {"db": "ok"}
    except Exception as e:
        return {"db": "connecting", "detail": str(e) or repr(e)}
