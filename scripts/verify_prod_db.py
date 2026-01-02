
import asyncio
import asyncpg
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import init_settings

async def verify_db():
    print("🔄 Loading settings...")
    try:
        settings = init_settings()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return

    print(f"✅ Settings Loaded.")
    print(f"   Server: {settings.POSTGRES_SERVER}")
    print(f"   Database: {settings.POSTGRES_DB}")
    print(f"   User: {settings.POSTGRES_USER}")
    print(f"   Schema Target: {settings.DB_SCHEMA}")
    print(f"   Environment: {settings.ENV_MODE}")

    # Build DSN
    dsn = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    
    # SSL Logic
    ssl_mode = False
    if settings.ENV_MODE != "dev" or "rds.amazonaws.com" in settings.POSTGRES_SERVER:
        ssl_mode = "require"
        print("🔒 SSL: require (Production/Staging/RDS)")
    else:
        print("🔓 SSL: disabled (Dev Mode)")

    print("\n⏳ Connecting to database...")
    try:
        conn = await asyncpg.connect(dsn, ssl=ssl_mode)
        print("✅ Connection Successful!")
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        return

    try:
        # Check Schema
        print(f"\n🔍 Checking if schema '{settings.DB_SCHEMA}' exists...")
        schema_exists = await conn.fetchval(
            "SELECT exists(select schema_name FROM information_schema.schemata WHERE schema_name = $1)",
            settings.DB_SCHEMA
        )
        
        if schema_exists:
            print(f"✅ Schema '{settings.DB_SCHEMA}' exists.")
        else:
            print(f"⚠️ Schema '{settings.DB_SCHEMA}' DOES NOT EXIST.")
        
        # Check Tables
        print(f"\n🔍 Listing tables in schema '{settings.DB_SCHEMA}':")
        rows = await conn.fetch(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = $1
            """,
            settings.DB_SCHEMA
        )
        
        if rows:
            print(f"Found {len(rows)} tables:")
            for row in rows:
                t_name = row['table_name']
                print(f" - {t_name}")
                
                # Check columns
                cols = await conn.fetch(
                    """
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_schema = $1 AND table_name = $2
                    """,
                    settings.DB_SCHEMA, t_name
                )
                col_names = [c['column_name'] for c in cols]
                if 'report_url' in col_names:
                    print(f"   ⚠️  Has 'report_url' column (Deprecated)")
                else:
                    print(f"   ✅  No 'report_url' column")
        else:
            print("⚠️ No tables found in this schema.")
            
        # Check if we can create tables (Permissions)
        # We won't actually create one, but just verifying connection read/write is good. 
        # But for now, listing is enough proof of access.

    except Exception as e:
        print(f"❌ Error during inspection: {e}")
    finally:
        await conn.close()
        print("\n🔌 Connection closed.")

if __name__ == "__main__":
    asyncio.run(verify_db())
