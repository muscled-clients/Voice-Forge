#!/usr/bin/env python
"""
Migrate VoiceForge Database to New Schema
Adds missing columns to existing tables
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def migrate_database():
    """Add missing columns to existing tables"""
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
        
        logger.info("Starting database migration...")
        
        # Add missing columns to users table
        migrations = [
            # User table columns
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS company VARCHAR(255)",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'user'",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT false",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(255)",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS password_reset_token VARCHAR(255)",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS password_reset_expires TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255)",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS subscription_status VARCHAR(50) DEFAULT 'inactive'",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP WITH TIME ZONE",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS api_calls_used INTEGER DEFAULT 0",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS api_calls_limit INTEGER DEFAULT 1000",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT false",
            "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS last_api_call TIMESTAMP WITH TIME ZONE",
            
            # Create indexes
            "CREATE INDEX IF NOT EXISTS idx_users_email ON voiceforge.users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_is_active ON voiceforge.users(is_active)",
            "CREATE INDEX IF NOT EXISTS idx_users_stripe_customer ON voiceforge.users(stripe_customer_id)",
            
            # Migrate password from metadata to password_hash column
            """
            UPDATE voiceforge.users 
            SET password_hash = metadata->>'password_hash'
            WHERE password_hash IS NULL 
            AND metadata->>'password_hash' IS NOT NULL
            """,
            
            # Migrate company from metadata
            """
            UPDATE voiceforge.users 
            SET company = metadata->>'company'
            WHERE company IS NULL 
            AND metadata->>'company' IS NOT NULL
            """
        ]
        
        for i, migration in enumerate(migrations, 1):
            try:
                cur.execute(migration)
                logger.info(f"✅ Migration {i}/{len(migrations)} completed")
            except Exception as e:
                logger.warning(f"⚠️ Migration {i} skipped or failed: {e}")
        
        # Verify the migration
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'voiceforge' 
            AND table_name = 'users'
            AND column_name = 'password_hash'
        """)
        
        if cur.fetchone():
            logger.info("✅ Database migration successful!")
            logger.info("✅ password_hash column exists")
            
            # Show updated schema
            cur.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'voiceforge' 
                AND table_name = 'users'
                ORDER BY ordinal_position
            """)
            
            logger.info("\nUpdated users table schema:")
            for row in cur.fetchall():
                logger.info(f"  - {row[0]}: {row[1]}")
        else:
            logger.error("❌ Migration failed - password_hash column not found")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔄 VoiceForge Database Migration")
    logger.info("=" * 50)
    
    success = migrate_database()
    
    if success:
        logger.info("\n✅ Migration completed successfully!")
        logger.info("You can now run the application with the updated schema.")
    else:
        logger.info("\n❌ Migration failed!")
        logger.info("Please check the error messages above.")
    
    sys.exit(0 if success else 1)