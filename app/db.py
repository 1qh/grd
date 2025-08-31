from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING

from sqlmodel import Session, SQLModel, create_engine

from app.config import get_settings

if TYPE_CHECKING:
  from collections.abc import Iterator

  from sqlalchemy.engine import Engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
  settings = get_settings()
  return create_engine(settings.DB_URL, echo=False, pool_pre_ping=True)


def init_db() -> None:
  engine = get_engine()
  SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
  with Session(get_engine()) as session:
    yield session
