"""Database session management."""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from ..common.config import get_settings

settings = get_settings()

Base = declarative_base()

_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            echo=settings.debug,
        )
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def _get_session_local():
    _get_engine()
    return _SessionLocal


engine = property(_get_engine)
SessionLocal = property(_get_session_local)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session.

    Yields:
        Database session

    Example:
        ```python
        from fastapi import Depends
        from database import get_db

        @app.get("/items")
        async def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
        ```
    """
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()


async def get_async_db():
    """Get async database session (placeholder for future async implementation)."""
    db = _get_session_local()()
    try:
        yield db
    finally:
        db.close()
