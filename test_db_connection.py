#!/usr/bin/env python3
"""
Test database connection using .env credentials
"""
import asyncio
import sys
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Add parent directory to path
sys.path.insert(0, '/app')

from app.config import settings

async def test_connection():
    print(f"Testing connection to: {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}")
    print(f"Database: {settings.POSTGRES_DB}")
    print(f"User: {settings.POSTGRES_USER}")
    print(f"SSL: require")
    print("-" * 50)
    
    # Create engine with SSL
    engine = create_async_engine(
        settings.SQLALCHEMY_DATABASE_URI,
        echo=False,
        pool_pre_ping=True,
        connect_args={
            "ssl": "require",
            "timeout": 60,
        }
    )
    
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT version()"))
            version = result.scalar()
            print("✅ Connection successful!")
            print(f"PostgreSQL version: {version}")
            
            # Check if table exists
            result = await conn.execute(text(
                "SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = 'crop_analysis_report')"
            ))
            table_exists = result.scalar()
            
            if table_exists:
                print("✅ Table 'crop_analysis_report' exists")
            else:
                print("⚠️  Table 'crop_analysis_report' does NOT exist - run migration script")
            
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        await engine.dispose()

if __name__ == "__main__":
    result = asyncio.run(test_connection())
    sys.exit(0 if result else 1)
