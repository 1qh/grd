from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Column, Enum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlmodel import Field, Relationship, SQLModel


def utcnow() -> datetime:
  return datetime.now(UTC)


class MessageRole(StrEnum):
  user = 'user'
  assistant = 'assistant'
  system = 'system'


class Conversation(SQLModel, table=True):
  __tablename__ = 'conversations'

  id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
  title: str | None = Field(default=None, nullable=True, index=True)
  model: str | None = Field(default=None, nullable=True, index=True)
  created_at: datetime = Field(default_factory=utcnow, nullable=False)
  updated_at: datetime = Field(default_factory=utcnow, nullable=False)

  messages: list[Message] = Relationship(sa_relationship=relationship('Message', back_populates='conversation'))


class Message(SQLModel, table=True):
  __tablename__ = 'messages'

  id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
  conversation_id: UUID = Field(foreign_key='conversations.id', index=True, nullable=False)
  role: MessageRole = Field(sa_column=Column(Enum(MessageRole, name='message_role'), nullable=False))
  attachments: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSONB, nullable=False))
  content: str = Field(default='', nullable=False)
  created_at: datetime = Field(default_factory=utcnow, nullable=False, index=True)

  conversation: Conversation | None = Relationship(
    sa_relationship=relationship('Conversation', back_populates='messages')
  )
