"""SQLAlchemy engine and session factory."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Session:
    """For use in scripts/CLI. FastAPI routes should use a dependency."""
    return SessionLocal()
