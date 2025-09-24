#!/usr/bin/env python
"""
Verify VoiceForge Database Tables
Check which tables were created in PostgreSQL
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def verify_database():
    """Check database tables and schemas"""
    try:
        # Build database URL
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "password")
        db_name = os.getenv("DB_NAME", "voiceforge_db")
        
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Check schemas
            result = conn.execute(text("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name IN ('voiceforge', 'analytics', 'monitoring')
                ORDER BY schema_name
            """))
            schemas = [row[0] for row in result]
            
            logger.info("=" * 60)
            logger.info("🗄️  DATABASE VERIFICATION REPORT")
            logger.info("=" * 60)
            logger.info(f"\n📦 Schemas Created: {len(schemas)}")
            for schema in schemas:
                logger.info(f"  ✅ {schema}")
            
            # Check tables in each schema
            table_query = text("""
                SELECT table_schema, table_name 
                FROM information_schema.tables 
                WHERE table_schema IN ('voiceforge', 'analytics', 'monitoring')
                AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
            """)
            
            result = conn.execute(table_query)
            tables = {}
            for row in result:
                schema = row[0]
                table = row[1]
                if schema not in tables:
                    tables[schema] = []
                tables[schema].append(table)
            
            total_tables = sum(len(t) for t in tables.values())
            logger.info(f"\n📊 Total Tables Created: {total_tables}")
            
            # List tables by schema
            for schema, table_list in sorted(tables.items()):
                logger.info(f"\n📁 Schema: {schema} ({len(table_list)} tables)")
                for table in table_list:
                    # Count rows
                    count_result = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}"))
                    count = count_result.scalar()
                    logger.info(f"  ✅ {table:30} ({count} rows)")
            
            # Check if plans have been inserted
            result = conn.execute(text("SELECT name, display_name, monthly_limit FROM voiceforge.plans ORDER BY monthly_limit"))
            plans = result.fetchall()
            
            if plans:
                logger.info(f"\n💳 Subscription Plans ({len(plans)} plans):")
                for plan in plans:
                    limit = "Unlimited" if plan[2] == -1 else f"{plan[2]:,} calls/month"
                    logger.info(f"  ✅ {plan[1]:20} ({plan[0]:10}) - {limit}")
            
            logger.info("\n" + "=" * 60)
            logger.info("✨ Database verification complete!")
            logger.info("📌 You can now view these tables in pgAdmin")
            logger.info("🚀 The portal sign-up/sign-in will use these tables")
            logger.info("=" * 60)
            
        engine.dispose()
        return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    verify_database()