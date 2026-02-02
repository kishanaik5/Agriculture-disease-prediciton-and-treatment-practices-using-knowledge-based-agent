"""Add multi-language columns to master_icons

Revision ID: add_master_icons_languages
Revises: 
Create Date: 2026-02-02 17:40:00

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_master_icons_languages'
down_revision = 'dpt2026012801'  # Latest migration (drop payment tables)
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add language columns to master_icons table"""
    
    # Add Tamil
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_ta VARCHAR(255)
    """)
    
    # Add Telugu
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_te VARCHAR(255)
    """)
    
    # Add Malayalam
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_ml VARCHAR(255)
    """)
    
    # Add Marathi
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_mr VARCHAR(255)
    """)
    
    # Add Gujarati
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_gu VARCHAR(255)
    """)
    
    # Add Bengali
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_bn VARCHAR(255)
    """)
    
    # Add Odia
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_or VARCHAR(255)
    """)
    
    # Add Punjabi
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_pa VARCHAR(255)
    """)
    
    # Add Urdu
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_ur VARCHAR(255)
    """)
    
    # Add Nepali
    op.execute("""
        ALTER TABLE kissan_cv.master_icons 
        ADD COLUMN IF NOT EXISTS name_ne VARCHAR(255)
    """)


def downgrade() -> None:
    """Remove language columns from master_icons table"""
    
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_ta")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_te")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_ml")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_mr")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_gu")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_bn")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_or")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_pa")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_ur")
    op.execute("ALTER TABLE kissan_cv.master_icons DROP COLUMN IF EXISTS name_ne")
