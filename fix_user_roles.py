#!/usr/bin/env python
"""
Fix user role values in database
Convert lowercase roles to uppercase to match SQLAlchemy enum
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

def fix_user_roles():
    """Fix user role values to match enum"""
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
        
        logger.info("Fixing user role values...")
        
        # First, let's see what roles we have
        cur.execute("""
            SELECT DISTINCT role 
            FROM voiceforge.users 
            WHERE role IS NOT NULL
        """)
        
        current_roles = cur.fetchall()
        logger.info(f"Current role values: {[r[0] for r in current_roles]}")
        
        # Update role values to uppercase
        updates = [
            ("UPDATE voiceforge.users SET role = 'USER' WHERE role = 'user' OR role IS NULL", "user -> USER"),
            ("UPDATE voiceforge.users SET role = 'ADMIN' WHERE role = 'admin'", "admin -> ADMIN"),
            ("UPDATE voiceforge.users SET role = 'DEVELOPER' WHERE role = 'developer'", "developer -> DEVELOPER"),
            ("UPDATE voiceforge.users SET role = 'ENTERPRISE' WHERE role = 'enterprise'", "enterprise -> ENTERPRISE"),
        ]
        
        for query, description in updates:
            cur.execute(query)
            affected = cur.rowcount
            if affected > 0:
                logger.info(f"✅ Updated {affected} rows: {description}")
        
        # Set default role for any null values
        cur.execute("UPDATE voiceforge.users SET role = 'USER' WHERE role IS NULL")
        if cur.rowcount > 0:
            logger.info(f"✅ Set default role 'USER' for {cur.rowcount} rows")
        
        # Verify the fix
        cur.execute("""
            SELECT id, email, role 
            FROM voiceforge.users 
            LIMIT 5
        """)
        
        logger.info("\nSample of updated users:")
        for row in cur.fetchall():
            logger.info(f"  - {row[1]}: {row[2]}")
        
        cur.close()
        conn.close()
        
        logger.info("\n✅ User roles fixed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to fix user roles: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Fixing User Roles")
    logger.info("=" * 50)
    
    success = fix_user_roles()
    
    if not success:
        logger.info("\n❌ Fix failed! Please check the error messages above.")
    
    import sys
    sys.exit(0 if success else 1)