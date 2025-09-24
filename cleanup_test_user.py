#!/usr/bin/env python
"""
Clean up test user for fresh registration
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

def cleanup_user(email):
    """Delete user and all related data"""
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
        
        logger.info(f"Cleaning up user: {email}")
        
        # Get user ID
        cur.execute("SELECT id FROM voiceforge.users WHERE email = %s", (email,))
        result = cur.fetchone()
        
        if result:
            user_id = result[0]
            logger.info(f"Found user ID: {user_id}")
            
            # Delete in order (due to foreign key constraints)
            tables_to_clean = [
                ("voiceforge.websocket_sessions", "user_id"),
                ("voiceforge.transcriptions", "user_id"),
                ("voiceforge.batch_jobs", "user_id"),
                ("voiceforge.developers", "user_id"),
                ("voiceforge.api_keys", "user_id"),
                ("voiceforge.user_subscriptions", "user_id"),
            ]
            
            for table, column in tables_to_clean:
                cur.execute(f"DELETE FROM {table} WHERE {column} = %s", (user_id,))
                count = cur.rowcount
                if count > 0:
                    logger.info(f"  ✅ Deleted {count} records from {table}")
            
            # Finally delete the user
            cur.execute("DELETE FROM voiceforge.users WHERE id = %s", (user_id,))
            logger.info(f"  ✅ Deleted user record")
            
            logger.info(f"✅ User {email} and all related data deleted successfully!")
        else:
            logger.info(f"User {email} not found")
        
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return False

if __name__ == "__main__":
    email = "syedawaishussain987@gmail.com"
    
    logger.info("🧹 User Cleanup Script")
    logger.info("=" * 60)
    
    response = input(f"Delete user {email} and all related data? (y/n): ")
    if response.lower() == 'y':
        success = cleanup_user(email)
        if success:
            logger.info("\n✨ Ready for fresh registration!")
    else:
        logger.info("Cleanup cancelled")
    
    import sys
    sys.exit(0)