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

# Add exception handlers for subscription errors
from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.subscription import (
    SubscriptionLimitExceeded,
    SubscriptionInvalid,
    SubscriptionServiceUnavailable
)

@app.exception_handler(SubscriptionLimitExceeded)
async def subscription_limit_exceeded_handler(request: Request, exc: SubscriptionLimitExceeded):
    """Handle subscription quota exceeded errors"""
    return JSONResponse(
        status_code=402,  # Payment Required
        content={
            "error": "subscription_limit_exceeded",
            "message": "You have reached your subscription limit. Please upgrade your plan or wait for the next billing cycle.",
            "details": str(exc) if str(exc) else None
        }
    )

@app.exception_handler(SubscriptionInvalid)
async def subscription_invalid_handler(request: Request, exc: SubscriptionInvalid):
    """Handle invalid subscription errors"""
    return JSONResponse(
        status_code=404,  # Not Found
        content={
            "error": "subscription_not_found",
            "message": "Subscription not found or has expired. Please check your subscription status.",
            "details": str(exc) if str(exc) else None
        }
    )

@app.exception_handler(SubscriptionServiceUnavailable)
async def subscription_service_unavailable_handler(request: Request, exc: SubscriptionServiceUnavailable):
    """Handle subscription service unavailability"""
    return JSONResponse(
        status_code=503,  # Service Unavailable
        content={
            "error": "subscription_service_unavailable",
            "message": "Subscription service is temporarily unavailable. Please try again later.",
            "details": str(exc) if str(exc) else None
        }
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
            # Create schema if it doesn't exist, then create tables
            async with engine.begin() as conn:
                # Create schema first
                await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {settings.DB_SCHEMA}"))
                print(f"✅ Schema '{settings.DB_SCHEMA}' ensured")
                
                # CRITICAL: Set search_path BEFORE creating tables
                # This ensures ALL tables (including SharedBackend's entities) 
                # are created in kissan_cv schema, not public
                await conn.execute(text(f"SET search_path TO {settings.DB_SCHEMA}, public"))
                # Note: No commit needed - the begin() context auto-commits on exit
                print(f"✅ Search path set to {settings.DB_SCHEMA}")
                
                # Now create all tables in the kissan_cv schema
                await conn.run_sync(BaseSchema.metadata.create_all)
                print(f"✅ Tables created in schema '{settings.DB_SCHEMA}'")
            
            # Verify connection
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            print(f"✅ Database connection successful and schema/tables verified.")
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
