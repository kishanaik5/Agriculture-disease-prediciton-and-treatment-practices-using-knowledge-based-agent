#!/usr/bin/env python3
"""
Run database migration from within Docker environment
"""
import asyncio
import sys
import re
sys.path.insert(0, '/app')

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text
from app.config import settings

def parse_sql_file(content):
    """Parse SQL file and return individual statements"""
    # Remove single-line comments
    lines = []
    for line in content.split('\n'):
        # Remove comment-only lines
        stripped = line.strip()
        if stripped.startswith('--'):
            continue
        # Remove inline comments
        if '--' in line:
            line = line.split('--')[0]
        lines.append(line)
    
    # Join and split by semicolons
    clean_content = '\n'.join(lines)
    statements = []
    
    for stmt in clean_content.split(';'):
        stmt = stmt.strip()
        if stmt:
            statements.append(stmt)
    
    return statements

async def run_migration():
    print("Running database migration...")
    print(f"Target: {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
    print(f"User: {settings.POSTGRES_USER}")
    print("-" * 50)
    
    engine = create_async_engine(
        settings.SQLALCHEMY_DATABASE_URI,
        echo=False,
        pool_pre_ping=True,
        connect_args={
            "ssl": "require",
            "timeout": 60,
        }
    )
    
    # Read migration SQL
    with open('/app/migrations/001_initial_schema.sql', 'r') as f:
        sql_content = f.read()
    
    statements = parse_sql_file(sql_content)
    print(f"Found {len(statements)} SQL statements to execute\n")
    
    try:
        async with engine.begin() as conn:
            for i, statement in enumerate(statements, 1):
                # Show first 60 chars of statement
                preview = statement[:60].replace('\n', ' ')
                if len(statement) > 60:
                    preview += "..."
                    
                print(f"{i}/{len(statements)}: {preview}")
                
                try:
                    await conn.execute(text(statement))
                    print("  ✅ Success")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "permission denied" in error_msg or "insufficient privilege" in error_msg:
                        print(f"  ❌ PERMISSION DENIED")
                        print("\n" + "=" * 50)
                        print("⚠️  User 'kissan_cv_dev' lacks CREATE privileges!")
                        print("\nYou need a database admin to run:")
                        print("  GRANT CREATE ON SCHEMA public TO kissan_cv_dev;")
                        print("=" * 50)
                        return False
                    else:
                        print(f"  ❌ Error: {e}")
                        raise
            
            print("\n" + "=" * 50)
            print("✅ Migration completed successfully!\n")
            
            # Verify table was created
            result = await conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = 'crop_analysis_report'"
            ))
            table = result.scalar()
            
            if table:
                print(f"✅ Verified: Table '{table}' exists")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        return False
    finally:
        await engine.dispose()

if __name__ == "__main__":
    result = asyncio.run(run_migration())
    sys.exit(0 if result else 1)
