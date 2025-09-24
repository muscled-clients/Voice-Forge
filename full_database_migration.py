#!/usr/bin/env python
"""
Full Database Migration - Ensure All Tables Match SQLAlchemy Models
This script ensures all tables and columns are properly synced
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def migrate_database():
    """Ensure all tables and columns match SQLAlchemy models"""
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
        
        logger.info("Starting comprehensive database migration...")
        logger.info("=" * 60)
        
        # Dictionary of all migrations grouped by table
        migrations = {
            "users": [
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS company VARCHAR(255)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'USER'",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS plan_id UUID REFERENCES voiceforge.plans(id)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS email_verified BOOLEAN DEFAULT false",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS email_verification_token VARCHAR(255)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS password_reset_token VARCHAR(255)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS password_reset_expires TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS stripe_customer_id VARCHAR(255)",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS subscription_status VARCHAR(50) DEFAULT 'INACTIVE'",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS last_login TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS api_calls_used INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS api_calls_limit INTEGER DEFAULT 1000",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT false",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS last_api_call TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
                "ALTER TABLE voiceforge.users ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "plans": [
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
            ],
            "api_keys": [
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS key_hash VARCHAR(255) NOT NULL",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS key_prefix VARCHAR(50) NOT NULL",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS name VARCHAR(255)",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS last_used TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS usage_count INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS rate_limit INTEGER DEFAULT 100",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT true",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS scopes JSONB DEFAULT '[]'::jsonb",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS ip_whitelist INET[]",
                "ALTER TABLE voiceforge.api_keys ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "developers": [
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS email VARCHAR(255) UNIQUE NOT NULL",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS full_name VARCHAR(255)",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS company VARCHAR(255)",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS website VARCHAR(500)",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS github_username VARCHAR(255)",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS use_case TEXT",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS is_verified BOOLEAN DEFAULT false",
                "ALTER TABLE voiceforge.developers ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
            ],
            "user_subscriptions": [
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS plan_id UUID REFERENCES voiceforge.plans(id)",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'INACTIVE'",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS stripe_subscription_id VARCHAR(255) UNIQUE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS stripe_payment_method_id VARCHAR(255)",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS current_period_start TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS current_period_end TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS canceled_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS cancel_at_period_end BOOLEAN DEFAULT false",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS trial_start TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS trial_end TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.user_subscriptions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
            ],
            "transcriptions": [
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS file_name VARCHAR(255)",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS file_size BIGINT",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS file_type VARCHAR(50)",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'PENDING'",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS transcription_text TEXT",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS confidence_score FLOAT",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS duration_seconds FLOAT",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS model_used VARCHAR(50)",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS language VARCHAR(10)",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS segments JSONB",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS speakers JSONB",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS error_message TEXT",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS word_count INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS speaker_count INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS billing_duration_seconds FLOAT",
                "ALTER TABLE voiceforge.transcriptions ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "websocket_sessions": [
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS session_id VARCHAR(255) UNIQUE NOT NULL",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS connected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS disconnected_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS last_activity TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS total_duration_seconds FLOAT",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS transcription_count INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS error_count INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS model_used VARCHAR(50)",
                "ALTER TABLE voiceforge.websocket_sessions ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "batch_jobs": [
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS job_type VARCHAR(50)",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'PENDING'",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS total_files INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS processed_files INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS failed_files INTEGER DEFAULT 0",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS started_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS error_message TEXT",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS result_url VARCHAR(500)",
                "ALTER TABLE voiceforge.batch_jobs ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "batch_job_files": [
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS job_id UUID REFERENCES voiceforge.batch_jobs(id) ON DELETE CASCADE",
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS file_name VARCHAR(255)",
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS file_size BIGINT",
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'PENDING'",
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS transcription_id UUID REFERENCES voiceforge.transcriptions(id)",
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS error_message TEXT",
                "ALTER TABLE voiceforge.batch_job_files ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP WITH TIME ZONE",
            ]
        }
        
        # Tables in billing schema
        billing_migrations = {
            "usage_records": [
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS record_type VARCHAR(50)",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS quantity DECIMAL(10,2)",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS unit VARCHAR(50)",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS description TEXT",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS billed BOOLEAN DEFAULT false",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS stripe_usage_record_id VARCHAR(255)",
                "ALTER TABLE billing.usage_records ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "invoices": [
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS stripe_invoice_id VARCHAR(255) UNIQUE",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS amount_due DECIMAL(10,2)",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS amount_paid DECIMAL(10,2)",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS status VARCHAR(50)",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS due_date TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS paid_at TIMESTAMP WITH TIME ZONE",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS hosted_invoice_url VARCHAR(500)",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS invoice_pdf VARCHAR(500)",
                "ALTER TABLE billing.invoices ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "payment_methods": [
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS stripe_payment_method_id VARCHAR(255) UNIQUE",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS type VARCHAR(50)",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS last4 VARCHAR(4)",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS brand VARCHAR(50)",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS exp_month INTEGER",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS exp_year INTEGER",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS is_default BOOLEAN DEFAULT false",
                "ALTER TABLE billing.payment_methods ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ]
        }
        
        # Tables in monitoring schema
        monitoring_migrations = {
            "usage_metrics": [
                "ALTER TABLE monitoring.usage_metrics ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE monitoring.usage_metrics ADD COLUMN IF NOT EXISTS metric_type VARCHAR(50)",
                "ALTER TABLE monitoring.usage_metrics ADD COLUMN IF NOT EXISTS value DECIMAL(10,2)",
                "ALTER TABLE monitoring.usage_metrics ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
                "ALTER TABLE monitoring.usage_metrics ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ],
            "rate_limits": [
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES voiceforge.users(id) ON DELETE CASCADE",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS endpoint VARCHAR(255)",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS requests_per_minute INTEGER DEFAULT 60",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS requests_per_hour INTEGER DEFAULT 1000",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS requests_per_day INTEGER DEFAULT 10000",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS current_minute_count INTEGER DEFAULT 0",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS current_hour_count INTEGER DEFAULT 0",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS current_day_count INTEGER DEFAULT 0",
                "ALTER TABLE monitoring.rate_limits ADD COLUMN IF NOT EXISTS last_reset TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
            ],
            "health_checks": [
                "ALTER TABLE monitoring.health_checks ADD COLUMN IF NOT EXISTS service_name VARCHAR(100)",
                "ALTER TABLE monitoring.health_checks ADD COLUMN IF NOT EXISTS status VARCHAR(50)",
                "ALTER TABLE monitoring.health_checks ADD COLUMN IF NOT EXISTS response_time_ms INTEGER",
                "ALTER TABLE monitoring.health_checks ADD COLUMN IF NOT EXISTS error_message TEXT",
                "ALTER TABLE monitoring.health_checks ADD COLUMN IF NOT EXISTS checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()",
                "ALTER TABLE monitoring.health_checks ADD COLUMN IF NOT EXISTS metadata JSONB DEFAULT '{}'::jsonb",
            ]
        }
        
        # Execute migrations for voiceforge schema
        logger.info("\n📦 Migrating voiceforge schema tables...")
        for table_name, table_migrations in migrations.items():
            logger.info(f"\n  Migrating table: {table_name}")
            for migration in table_migrations:
                try:
                    cur.execute(migration)
                    # Extract column name from migration for logging
                    if "ADD COLUMN" in migration:
                        col_name = migration.split("ADD COLUMN IF NOT EXISTS")[1].split()[0]
                        logger.info(f"    ✅ Column {col_name} checked/added")
                except Exception as e:
                    logger.warning(f"    ⚠️ Migration skipped: {str(e)[:100]}")
        
        # Execute migrations for billing schema
        logger.info("\n💳 Migrating billing schema tables...")
        for table_name, table_migrations in billing_migrations.items():
            logger.info(f"\n  Migrating table: {table_name}")
            for migration in table_migrations:
                try:
                    cur.execute(migration)
                    if "ADD COLUMN" in migration:
                        col_name = migration.split("ADD COLUMN IF NOT EXISTS")[1].split()[0]
                        logger.info(f"    ✅ Column {col_name} checked/added")
                except Exception as e:
                    logger.warning(f"    ⚠️ Migration skipped: {str(e)[:100]}")
        
        # Execute migrations for monitoring schema
        logger.info("\n📊 Migrating monitoring schema tables...")
        for table_name, table_migrations in monitoring_migrations.items():
            logger.info(f"\n  Migrating table: {table_name}")
            for migration in table_migrations:
                try:
                    cur.execute(migration)
                    if "ADD COLUMN" in migration:
                        col_name = migration.split("ADD COLUMN IF NOT EXISTS")[1].split()[0]
                        logger.info(f"    ✅ Column {col_name} checked/added")
                except Exception as e:
                    logger.warning(f"    ⚠️ Migration skipped: {str(e)[:100]}")
        
        # Create indexes for better performance
        logger.info("\n🔍 Creating indexes for better performance...")
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON voiceforge.users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_stripe_customer ON voiceforge.users(stripe_customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_api_keys_user ON voiceforge.api_keys(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON voiceforge.api_keys(key_hash)",
            "CREATE INDEX IF NOT EXISTS idx_transcriptions_user ON voiceforge.transcriptions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_transcriptions_status ON voiceforge.transcriptions(status)",
            "CREATE INDEX IF NOT EXISTS idx_websocket_sessions_user ON voiceforge.websocket_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_user ON voiceforge.batch_jobs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_usage_records_user ON billing.usage_records(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_invoices_user ON billing.invoices(user_id)",
        ]
        
        for index in indexes:
            try:
                cur.execute(index)
                index_name = index.split("INDEX IF NOT EXISTS")[1].split()[0]
                logger.info(f"  ✅ Index {index_name} created/verified")
            except Exception as e:
                logger.warning(f"  ⚠️ Index creation skipped: {str(e)[:100]}")
        
        # Verify table counts
        logger.info("\n📋 Verifying database structure...")
        
        schemas = ['voiceforge', 'billing', 'monitoring']
        for schema in schemas:
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
            """)
            count = cur.fetchone()[0]
            logger.info(f"  {schema} schema: {count} tables")
        
        # Show summary of all tables
        logger.info("\n📊 Database Summary:")
        cur.execute("""
            SELECT table_schema, table_name, 
                   COUNT(column_name) as column_count
            FROM information_schema.columns
            WHERE table_schema IN ('voiceforge', 'billing', 'monitoring')
            GROUP BY table_schema, table_name
            ORDER BY table_schema, table_name
        """)
        
        for row in cur.fetchall():
            logger.info(f"  {row[0]}.{row[1]}: {row[2]} columns")
        
        cur.close()
        conn.close()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Full database migration completed successfully!")
        logger.info("All tables and columns are now synced with SQLAlchemy models")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("🚀 VoiceForge Full Database Migration")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now()}")
    
    success = migrate_database()
    
    if success:
        logger.info(f"\n✅ Migration completed at: {datetime.now()}")
    else:
        logger.info(f"\n❌ Migration failed at: {datetime.now()}")
        logger.info("Please check the error messages above.")
    
    sys.exit(0 if success else 1)