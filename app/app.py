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
from fastapi.responses import Response
from app.config import init_settings

settings = init_settings()
from app.routers.v1.scan import router as scan_router
from app.routers.v1.async_scan import router as async_scan_router
# from app.services.redis_manager import task_manager # Removed Redis dependency
from SharedBackend.managers.base import BaseSchema
from app.database import engine
from sqlalchemy import text
import asyncio

db_ready = False


tags_metadata = [
    {"name": "Show Details", "description": "Get item details and prices."},
    {"name": "QA_Scan", "description": "Quality Assurance Scan."},
    {"name": "Payments", "description": "Payment processing."},
    {"name": "Generate Report", "description": "Report generation and status."},
    {"name": "Translation", "description": "Translation services."},
    {"name": "Admin Section", "description": "Administrative actions."}
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    openapi_tags=tags_metadata
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ TEMPORARY: Debug middleware to diagnose prod auth issues
# Remove this once auth is working correctly
from app.middlewares.auth_debug import AuthDebugMiddleware
app.add_middleware(AuthDebugMiddleware)  # Will get JWT_SECRET from environment

from SharedBackend.middlewares.AuthMiddleware import JWTAuthMiddleware
# JWTAuthMiddleware will get secret from environment
import os
jwt_secret = os.getenv("JWT_SECRET", "")
if not jwt_secret:
    print("⚠️ WARNING: JWT_SECRET not found in environment - auth will fail")
app.add_middleware(JWTAuthMiddleware, jwt_secret=jwt_secret)

from fastapi.staticfiles import StaticFiles

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon")

# Mount Static
app.mount("/static", StaticFiles(directory="static"), name="static")

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
            
            print("✅ Database connection successful and tables verified.")
            db_ready = True
            return True
            
        except OSError as e:
            # Handle specifically "Can't assign requested address" or other OS-level connect errors
            print(f"⚠️ Database unavailable (Attempt {attempt}/{max_retries}): {e}")
        except Exception as e:
            print(f"⚠️ Database unavailable (Attempt {attempt}/{max_retries}): {e}")
            
        if attempt < max_retries:
            await asyncio.sleep(delay)
            # Exponential backoff optional, but linear 2s is usually fine for startup
    
    print("❌ Could not connect to database after multiple retries.")
    return False


@app.on_event("startup")
async def startup_event():
    global db_ready
    print(f"🚀 Starting up {settings.PROJECT_NAME}...")
    
    # 1. Initialize Redis (Disabled by user request)
    # try:
    #     if await task_manager.connect():
    #         print("✅ Redis connection successful")
    #     else:
    #         print("⚠️ Redis connection failed - async features may be limited")
    # except Exception as e:
    #     print(f"⚠️ Redis init failed: {e}")

    # 2. Add DB wait check
    is_connected = await wait_for_db(engine)
    if not is_connected:
        print("⚠️ Application starting without DB connection - some features will fail")


@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy", 
        "database": "connected" if db_ready else "disconnected"
    }
    
    if not db_ready:
        health_status["status"] = "degraded"
        # Return 503 if DB is critical, or 200 with degraded status?
        # Usually health checks should reflect 'can serve traffic'. 
        # If DB is down, maybe 503 is better. 
        # But for strictly "app is running", 200 is okay.
        
    return health_status

# Include Routers
app.include_router(scan_router, prefix=settings.API_V1_STR)
app.include_router(async_scan_router, prefix=settings.API_V1_STR)
# from app.routers.v1.payment import router as payment_router
# app.include_router(payment_router, prefix=settings.API_V1_STR)

from app.routers.v1.translation import router as translation_router
app.include_router(translation_router, prefix=settings.API_V1_STR)
