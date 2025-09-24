#!/usr/bin/env python
"""
Initialize VoiceForge Database
Creates all tables and schemas in PostgreSQL
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from src.database.connection import init_db
from src.database.models import Base
from sqlalchemy import create_engine, text
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_database():
    """Create database and schemas if they don't exist"""
    try:
        # Build database URL
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "password")
        db_name = os.getenv("DB_NAME", "voiceforge_db")
        
        # First connect to postgres database to create voiceforge_db if needed
        postgres_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/postgres"
        engine = create_engine(postgres_url)
        
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = :dbname"
            ), {"dbname": db_name})
            
            if not result.scalar():
                # Create database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                logger.info(f"✅ Created database: {db_name}")
                conn.commit()
            else:
                logger.info(f"✅ Database already exists: {db_name}")
        
        engine.dispose()
        
        # Now connect to the actual database
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url)
        
        # Create schemas
        with engine.connect() as conn:
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS voiceforge"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS analytics"))
            conn.execute(text("CREATE SCHEMA IF NOT EXISTS monitoring"))
            conn.commit()
            logger.info("✅ Created database schemas")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Created all database tables")
        
        # Create initial data
        with engine.connect() as conn:
            # Check if plans exist
            result = conn.execute(text(
                "SELECT COUNT(*) FROM voiceforge.plans"
            ))
            count = result.scalar()
            
            if count == 0:
                # Insert default plans
                import uuid
                plans = [
                    {
                        'id': str(uuid.uuid4()),
                        'name': 'free',
                        'display_name': 'Free Plan',
                        'description': 'Perfect for getting started',
                        'monthly_limit': 1000,
                        'price_monthly': 0,
                        'price_yearly': 0,
                        'features': ["1,000 API calls/month", "Basic transcription", "Community support"]
                    },
                    {
                        'id': str(uuid.uuid4()),
                        'name': 'pro',
                        'display_name': 'Pro Plan',
                        'description': 'For professional developers',
                        'monthly_limit': 10000,
                        'price_monthly': 29.99,
                        'price_yearly': 299.99,
                        'features': ["10,000 API calls/month", "Advanced features", "Priority support", "Speaker diarization", "Batch processing"]
                    },
                    {
                        'id': str(uuid.uuid4()),
                        'name': 'business',
                        'display_name': 'Business Plan',
                        'description': 'For teams and businesses',
                        'monthly_limit': 50000,
                        'price_monthly': 99.99,
                        'price_yearly': 999.99,
                        'features': ["50,000 API calls/month", "All Pro features", "Dedicated support", "Custom webhooks", "99.9% SLA"]
                    },
                    {
                        'id': str(uuid.uuid4()),
                        'name': 'enterprise',
                        'display_name': 'Enterprise',
                        'description': 'Custom solutions',
                        'monthly_limit': -1,
                        'price_monthly': 0,
                        'price_yearly': 0,
                        'features': ["Unlimited API calls", "Custom features", "24/7 phone support", "On-premise option", "Custom contracts"]
                    }
                ]
                
                import json
                for plan in plans:
                    conn.execute(text("""
                        INSERT INTO voiceforge.plans (id, name, display_name, description, monthly_limit, price_monthly, price_yearly, features)
                        VALUES (:id, :name, :display_name, :description, :monthly_limit, :price_monthly, :price_yearly, :features::json)
                    """), {
                        'id': plan['id'],
                        'name': plan['name'],
                        'display_name': plan['display_name'],
                        'description': plan['description'],
                        'monthly_limit': plan['monthly_limit'],
                        'price_monthly': plan['price_monthly'],
                        'price_yearly': plan['price_yearly'] if plan['price_yearly'] else plan['price_monthly'] * 10,
                        'features': json.dumps(plan['features'])
                    })
                conn.commit()
                logger.info("✅ Created default subscription plans")
        
        engine.dispose()
        
        logger.info("✨ Database initialization complete!")
        logger.info("📊 You can now see the tables in pgAdmin:")
        logger.info("   - Schema: voiceforge (main tables)")
        logger.info("   - Schema: analytics (usage metrics)")
        logger.info("   - Schema: monitoring (rate limits, health)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = create_database()
    sys.exit(0 if success else 1)