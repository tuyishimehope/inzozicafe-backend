from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    # SQL parameters hold user text; show them in errors only in development.
    hide_parameters=settings.ENVIRONMENT.strip().lower() != "development",
    pool_pre_ping=True,  # replace connections the server closed (restart, idle timeout)
)
SessionLocal = async_sessionmaker(engine, autoflush=False, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_db)]
