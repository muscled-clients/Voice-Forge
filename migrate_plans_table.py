#!/usr/bin/env python
"""
Migrate Plans Table - Add Missing Columns
Adds missing columns to the plans table
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def migrate_plans_table():
    """Add missing columns to plans table"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "voiceforge_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "Gateway123")
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        logger.info("Migrating plans table...")
        
        # Add missing columns to plans table
        migrations = [
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS display_name VARCHAR(100)",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS description TEXT",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS monthly_limit INTEGER DEFAULT 0",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS price_monthly DECIMAL(10,2) DEFAULT 0.00",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS price_yearly DECIMAL(10,2) DEFAULT 0.00",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS features JSONB DEFAULT '{}'::jsonb",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS stripe_price_id VARCHAR(255)",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS stripe_product_id VARCHAR(255)",
            "ALTER TABLE voiceforge.plans ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
        ]
        
        for i, migration in enumerate(migrations, 1):
            try:
                cur.execute(migration)
                logger.info(f"✅ Migration {i}/{len(migrations)} completed")
            except Exception as e:
                logger.warning(f"⚠️ Migration {i} skipped or failed: {e}")
        
        # Check if free plan exists, if not create it
        cur.execute("SELECT id FROM voiceforge.plans WHERE name = 'free'")
        if not cur.fetchone():
            logger.info("Creating free plan...")
            cur.execute("""
                INSERT INTO voiceforge.plans (
                    id, name, display_name, description, monthly_limit, 
                    price_monthly, price_yearly, features, is_active
                ) VALUES (
                    gen_random_uuid(),
                    'free',
                    'Free Plan',
                    'Basic plan for getting started',
                    1000,
                    0.00,
                    0.00,
                    '{"api_calls": 1000, "rate_limit": 10, "support": "community"}'::jsonb,
                    true
                )
            """)
            logger.info("✅ Free plan created")
        
        # Update existing plans with default values if needed
        cur.execute("""
            UPDATE voiceforge.plans 
            SET display_name = COALESCE(display_name, INITCAP(name)),
                description = COALESCE(description, 'Plan ' || name),
                monthly_limit = COALESCE(monthly_limit, 1000),
                features = COALESCE(features, '{}'::jsonb)
            WHERE display_name IS NULL OR description IS NULL
        """)
        
        # Verify the migration
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'voiceforge' 
            AND table_name = 'plans'
            ORDER BY ordinal_position
        """)
        
        logger.info("\nUpdated plans table schema:")
        for row in cur.fetchall():
            logger.info(f"  - {row[0]}: {row[1]}")
        
        # Show existing plans
        cur.execute("""
            SELECT name, display_name, monthly_limit, price_monthly 
            FROM voiceforge.plans
        """)
        
        logger.info("\nExisting plans:")
        for row in cur.fetchall():
            logger.info(f"  - {row[0]}: {row[1]} ({row[2]} calls, ${row[3]}/mo)")
        
        cur.close()
        conn.close()
        
        logger.info("\n✅ Plans table migration successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔄 Plans Table Migration")
    logger.info("=" * 50)
    
    success = migrate_plans_table()
    
    if success:
        logger.info("\n✅ Migration completed successfully!")
    else:
        logger.info("\n❌ Migration failed! Please check the error messages above.")
    
    sys.exit(0 if success else 1)