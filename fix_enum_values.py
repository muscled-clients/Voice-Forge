#!/usr/bin/env python
"""
Fix all enum values in database to uppercase
Ensures consistency with SQLAlchemy enum definitions
"""

import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def fix_enum_values():
    """Fix all enum values to uppercase"""
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
        
        logger.info("Fixing enum values in database...")
        
        # Fix subscription_status values
        subscription_updates = [
            ("UPDATE voiceforge.users SET subscription_status = 'INACTIVE' WHERE subscription_status = 'inactive' OR subscription_status IS NULL", "inactive -> INACTIVE"),
            ("UPDATE voiceforge.users SET subscription_status = 'ACTIVE' WHERE subscription_status = 'active'", "active -> ACTIVE"),
            ("UPDATE voiceforge.users SET subscription_status = 'CANCELED' WHERE subscription_status = 'canceled'", "canceled -> CANCELED"),
            ("UPDATE voiceforge.users SET subscription_status = 'PAST_DUE' WHERE subscription_status = 'past_due'", "past_due -> PAST_DUE"),
            ("UPDATE voiceforge.users SET subscription_status = 'UNPAID' WHERE subscription_status = 'unpaid'", "unpaid -> UNPAID"),
            ("UPDATE voiceforge.users SET subscription_status = 'TRIALING' WHERE subscription_status = 'trialing'", "trialing -> TRIALING"),
        ]
        
        logger.info("\nFixing subscription_status values:")
        for query, description in subscription_updates:
            cur.execute(query)
            affected = cur.rowcount
            if affected > 0:
                logger.info(f"  ✅ Updated {affected} rows: {description}")
        
        # Set default subscription_status for null values
        cur.execute("UPDATE voiceforge.users SET subscription_status = 'INACTIVE' WHERE subscription_status IS NULL")
        if cur.rowcount > 0:
            logger.info(f"  ✅ Set default 'INACTIVE' for {cur.rowcount} rows with NULL subscription_status")
        
        # If there are transcriptions table, fix those enums too
        try:
            cur.execute("SELECT 1 FROM voiceforge.transcriptions LIMIT 1")
            
            # Fix transcription status values
            transcription_updates = [
                ("UPDATE voiceforge.transcriptions SET status = 'PENDING' WHERE status = 'pending'", "pending -> PENDING"),
                ("UPDATE voiceforge.transcriptions SET status = 'PROCESSING' WHERE status = 'processing'", "processing -> PROCESSING"),
                ("UPDATE voiceforge.transcriptions SET status = 'COMPLETED' WHERE status = 'completed'", "completed -> COMPLETED"),
                ("UPDATE voiceforge.transcriptions SET status = 'FAILED' WHERE status = 'failed'", "failed -> FAILED"),
                ("UPDATE voiceforge.transcriptions SET status = 'CANCELED' WHERE status = 'canceled'", "canceled -> CANCELED"),
            ]
            
            logger.info("\nFixing transcription status values:")
            for query, description in transcription_updates:
                cur.execute(query)
                affected = cur.rowcount
                if affected > 0:
                    logger.info(f"  ✅ Updated {affected} rows: {description}")
        except:
            logger.info("  ℹ️ No transcriptions table found or no data to update")
        
        # Verify the fixes
        logger.info("\nVerifying updated values:")
        
        # Check users table
        cur.execute("""
            SELECT email, role, subscription_status 
            FROM voiceforge.users 
            LIMIT 5
        """)
        
        logger.info("\nSample of updated users:")
        for row in cur.fetchall():
            logger.info(f"  - {row[0]}: role={row[1]}, subscription={row[2]}")
        
        cur.close()
        conn.close()
        
        logger.info("\n✅ All enum values fixed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix enum values: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Fixing Database Enum Values")
    logger.info("=" * 50)
    
    success = fix_enum_values()
    
    if not success:
        logger.info("\n❌ Fix failed! Please check the error messages above.")
    
    import sys
    sys.exit(0 if success else 1)