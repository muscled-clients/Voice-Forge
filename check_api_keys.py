#!/usr/bin/env python
"""
Check API Keys in Database
Diagnostic script to check API key status for users
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

def check_api_keys():
    """Check API keys for all users"""
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
        
        logger.info("Checking users and their API keys...")
        logger.info("=" * 60)
        
        # Check all users
        cur.execute("""
            SELECT u.id, u.email, u.full_name, u.role, u.plan_id, u.created_at
            FROM voiceforge.users u
            ORDER BY u.created_at DESC
        """)
        
        users = cur.fetchall()
        logger.info(f"\n📊 Found {len(users)} users:")
        
        for user in users:
            user_id, email, full_name, role, plan_id, created_at = user
            logger.info(f"\n👤 User: {email}")
            logger.info(f"   ID: {user_id}")
            logger.info(f"   Name: {full_name}")
            logger.info(f"   Role: {role}")
            logger.info(f"   Plan ID: {plan_id}")
            logger.info(f"   Created: {created_at}")
            
            # Check API keys for this user
            cur.execute("""
                SELECT id, key_prefix, name, is_active, created_at, last_used
                FROM voiceforge.api_keys
                WHERE user_id = %s
            """, (user_id,))
            
            api_keys = cur.fetchall()
            if api_keys:
                logger.info(f"   🔑 API Keys ({len(api_keys)}):")
                for key in api_keys:
                    key_id, prefix, name, is_active, created, last_used = key
                    logger.info(f"      - {name}: {prefix}... (Active: {is_active}, Created: {created})")
            else:
                logger.info(f"   ⚠️ No API keys found!")
        
        # Check plans
        logger.info("\n" + "=" * 60)
        logger.info("📋 Available Plans:")
        cur.execute("""
            SELECT id, name, display_name, monthly_limit, price_monthly
            FROM voiceforge.plans
            WHERE is_active = true
        """)
        
        plans = cur.fetchall()
        for plan in plans:
            plan_id, name, display_name, limit, price = plan
            logger.info(f"   {plan_id}: {display_name} ({name}) - {limit} calls/month, ${price}")
        
        # Check for orphaned API keys
        logger.info("\n" + "=" * 60)
        logger.info("🔍 Checking for issues...")
        
        cur.execute("""
            SELECT COUNT(*) FROM voiceforge.api_keys
            WHERE user_id NOT IN (SELECT id FROM voiceforge.users)
        """)
        orphaned = cur.fetchone()[0]
        if orphaned > 0:
            logger.warning(f"   ⚠️ Found {orphaned} orphaned API keys!")
        else:
            logger.info(f"   ✅ No orphaned API keys")
        
        # Check for users without plan
        cur.execute("""
            SELECT COUNT(*) FROM voiceforge.users
            WHERE plan_id IS NULL OR 
                  plan_id NOT IN (SELECT id FROM voiceforge.plans)
        """)
        no_plan = cur.fetchone()[0]
        if no_plan > 0:
            logger.warning(f"   ⚠️ Found {no_plan} users without valid plan!")
        else:
            logger.info(f"   ✅ All users have valid plans")
        
        cur.close()
        conn.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Check completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Check failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("🔍 API Key Diagnostic Check")
    logger.info("=" * 60)
    
    success = check_api_keys()
    
    if not success:
        logger.info("\n❌ Check failed! Please review the errors above.")
    
    import sys
    sys.exit(0 if success else 1)