"""
Update master_icons table with multi-language data from CSV

This script:
1. Reads master_icons_updated_with_languages.csv
2. Updates existing records by matching category_id
3. Preserves existing data (uid, created_at, etc.)
"""

import asyncio
import csv
import sys
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Add parent directory to path
sys.path.insert(0, '/app')

from app.database import SessionLocal
from app.models.scan import MasterIcon


async def update_master_icons():
    """Update master_icons table from CSV file"""
    
    csv_file = 'master_icons_updated_with_languages.csv'
    
    print(f"📂 Reading CSV file: {csv_file}")
    
    # Read CSV
    updates = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            updates.append({
                'category_id': row['category_id'],
                'category_type': row['category_type'],
                'url': row['url'],
                'name_en': row['name_en'],
                'name_kn': row['name_kn'],
                'name_hn': row['name_hn'],
                'name_ta': row['name_ta'],
                'name_te': row['name_te'],
                'name_ml': row['name_ml'],
                'name_mr': row['name_mr'],
                'name_gu': row['name_gu'],
                'name_bn': row['name_bn'],
                'name_or': row['name_or'],
                'name_pa': row.get('name_pa', ''),
                'name_ur': row.get('name_ur', ''),
                'name_ne': row.get('name_ne', ''),
            })
    
    print(f"📊 Found {len(updates)} records in CSV")
    
    # Update database
    async with SessionLocal() as db:
        updated_count = 0
        not_found_count = 0
        
        for update_data in updates:
            category_id = update_data['category_id']
            
            # Check if record exists
            stmt = select(MasterIcon).where(MasterIcon.category_id == category_id)
            result = await db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing record
                stmt = (
                    update(MasterIcon)
                    .where(MasterIcon.category_id == category_id)
                    .values(
                        category_type=update_data['category_type'],
                        url=update_data['url'],
                        name_en=update_data['name_en'],
                        name_kn=update_data['name_kn'],
                        name_hn=update_data['name_hn'],
                        name_ta=update_data['name_ta'],
                        name_te=update_data['name_te'],
                        name_ml=update_data['name_ml'],
                        name_mr=update_data['name_mr'],
                        name_gu=update_data['name_gu'],
                        name_bn=update_data['name_bn'],
                        name_or=update_data['name_or'],
                        name_pa=update_data['name_pa'],
                        name_ur=update_data['name_ur'],
                        name_ne=update_data['name_ne'],
                    )
                )
                await db.execute(stmt)
                updated_count += 1
                print(f"✅ Updated: {category_id} - {update_data['name_en']}")
            else:
                # Create new record
                new_icon = MasterIcon(**update_data)
                db.add(new_icon)
                updated_count += 1
                print(f"➕ Created: {category_id} - {update_data['name_en']}")
        
        await db.commit()
        
        print(f"\n✨ Update complete!")
        print(f"   Updated/Created: {updated_count}")
        print(f"   Not found: {not_found_count}")


if __name__ == "__main__":
    print("🚀 Starting master_icons update...")
    asyncio.run(update_master_icons())
    print("✅ Done!")
