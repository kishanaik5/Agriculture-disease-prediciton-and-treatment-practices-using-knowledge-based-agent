import sys
import os
from pathlib import Path

# Robustly set up PYTHONPATH to include SharedBackend
# This ensures it works in Docker, Local, and custom Deployments
try:
    current_dir = Path(__file__).resolve().parent.parent # /app
    shared_backend_src = current_dir / "SharedBackend" / "src"
    
    if shared_backend_src.exists():
        sys.path.insert(0, str(shared_backend_src))
        print(f"✅ Added {shared_backend_src} to sys.path")
    else:
        print(f"⚠️  SharedBackend path not found at: {shared_backend_src}")
except Exception as e:
    print(f"⚠️  Error setting up path: {e}")

from fastapi import FastAPI
from app.config import init_settings

settings = init_settings()
from app.routers.v1.scan import router as scan_router
from app.routers.v1.async_scan import router as async_scan_router
from app.services.redis_manager import task_manager
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
    # Initialize Redis connection
    await task_manager.connect()

@app.on_event("shutdown")
async def shutdown():
    # Clean up Redis connection
    await task_manager.close()

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app.include_router(scan_router, prefix=settings.API_V1_STR, tags=["scan"])
app.include_router(async_scan_router, prefix=settings.API_V1_STR, tags=["scan-async"])

# Only mount static files if directory exists
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    async def read_index():
        return FileResponse('static/index.html')
else:
    @app.get("/")
    async def read_index():
        return {"message": "CV Service API is running"}

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
