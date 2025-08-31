from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

  OPENAI_API_KEY: str
  OPENAI_BASE_URL: str
  DB_URL: str
  ENV: Literal['dev', 'prod'] | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
  return Settings()
