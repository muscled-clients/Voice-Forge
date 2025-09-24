"""
VoiceForge Database Connection Management
Using SQLAlchemy for ORM-based database operations
"""
import os
import logging
from typing import Optional, Generator
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnection:
    """Database connection manager using SQLAlchemy"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.session = None
        self.initialize()
    
    def initialize(self):
        """Initialize database connection"""
        try:
            # Build database URL from environment variables
            db_url = self._build_database_url()
            
            # Create engine with connection pooling
            self.engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=40,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                future=True  # Use SQLAlchemy 2.0 style
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
                expire_on_commit=False
            )
            
            # Create scoped session for thread safety
            self.session = scoped_session(self.SessionLocal)
            
            logger.info("✅ Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {e}")
            raise
    
    def _build_database_url(self) -> str:
        """Build database URL from environment variables"""
        db_user = os.getenv("DB_USER", "voiceforge")
        db_password = os.getenv("DB_PASSWORD", "password")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "voiceforge")
        
        # Support for DATABASE_URL (common in cloud deployments)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Handle Heroku-style postgres:// URLs
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)
            
            # Remove any query parameters that aren't valid for PostgreSQL
            # These might be application-level parameters
            if "?" in database_url:
                base_url = database_url.split("?")[0]
                logger.info(f"Stripped query parameters from DATABASE_URL")
                return base_url
            
            return database_url
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    @contextmanager
    def get_db(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def close(self):
        """Close database connection"""
        if self.session:
            self.session.remove()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def create_schemas(self):
        """Create database schemas if they don't exist"""
        try:
            with self.engine.connect() as conn:
                # Create schemas
                conn.execute("CREATE SCHEMA IF NOT EXISTS voiceforge")
                conn.execute("CREATE SCHEMA IF NOT EXISTS analytics")
                conn.execute("CREATE SCHEMA IF NOT EXISTS monitoring")
                conn.commit()
                logger.info("✅ Database schemas created successfully")
        except Exception as e:
            logger.error(f"Failed to create schemas: {e}")
            raise
    
    def init_database(self):
        """Initialize database with all tables"""
        try:
            from src.database.models import Base
            
            # Create schemas first
            self.create_schemas()
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("✅ Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

# Global database connection instance
db_connection: Optional[DatabaseConnection] = None

def init_db() -> DatabaseConnection:
    """Initialize global database connection"""
    global db_connection
    if not db_connection:
        db_connection = DatabaseConnection()
    return db_connection

def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection"""
    if not db_connection:
        init_db()
    
    with db_connection.get_db() as session:
        yield session

def get_db_session() -> Session:
    """Get a new database session"""
    if not db_connection:
        init_db()
    return db_connection.get_session()

def close_db():
    """Close database connection"""
    global db_connection
    if db_connection:
        db_connection.close()
        db_connection = None

# FastAPI dependency
async def get_async_db():
    """Async database session for FastAPI"""
    session = get_db_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()