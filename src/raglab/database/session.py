"""Async SQLAlchemy engine and session construction."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(dsn: str) -> AsyncEngine:
    """Create the application engine with stale-connection detection."""
    return create_async_engine(dsn, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create sessions that retain loaded state after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
