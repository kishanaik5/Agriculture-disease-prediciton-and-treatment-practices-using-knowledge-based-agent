#!/usr/bin/env python3
"""
Run Alembic migration using Python
"""
import sys
sys.path.insert(0, '/app')

from alembic.config import Config
from alembic import command

# Configure Alembic
alembic_cfg = Config("/app/alembic.ini")

try:
    print("Running Alembic migrations...")
    command.upgrade(alembic_cfg, "head")
    print("✅ Migrations completed successfully!")
except Exception as e:
    error_msg = str(e).lower()
    print(f"❌ Migration failed: {e}")
    
    if "permission denied" in error_msg or "insufficient privilege" in error_msg:
        print("\n⚠️  PERMISSION ERROR!")
        print("User 'kissan_cv_dev' does not have CREATE privileges.")
        print("\nAlembic also needs CREATE privileges - same issue as before.")
        sys.exit(1)
    
    raise
