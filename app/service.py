from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlmodel import select

from app.db import get_session
from app.models import Conversation, Message, MessageRole, utcnow

if TYPE_CHECKING:
  from uuid import UUID


@dataclass
class Attachment:
  type: str
  path: str
  mime: str


def create_conversation(model: str, title: str | None = None) -> Conversation:
  with get_session() as s:
    conv = Conversation(title=title, model=model)
    s.add(conv)
    s.commit()
    s.refresh(conv)
    return conv


def list_conversations(limit: int = 100) -> list[Conversation]:
  with get_session() as s:
    return s.exec(select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit)).all()


def get_conversation(conversation_id: UUID) -> Conversation | None:
  with get_session() as s:
    conv = s.get(Conversation, conversation_id)
    if not conv:
      return None
    return conv


def rename_conversation(conversation_id: UUID, title: str) -> bool:
  with get_session() as s:
    conv = s.get(Conversation, conversation_id)
    if not conv:
      return False
    conv.title = title
    s.add(conv)
    s.commit()
    return True


def add_message(
  conversation_id: UUID,
  role: MessageRole,
  content: str,
  attachments: list[dict[str, Any]] | None = None,
) -> Message:
  with get_session() as s:
    msg = Message(
      conversation_id=conversation_id,
      role=role,
      content=content,
      attachments=attachments or [],
    )
    s.add(msg)
    conv = s.get(Conversation, conversation_id)
    if conv:
      conv.updated_at = utcnow()
      s.add(conv)
    s.commit()
    s.refresh(msg)
    return msg


def delete_last_assistant_message(conversation_id: UUID) -> bool:
  with get_session() as s:
    msg = s.exec(
      select(Message)
      .where(Message.conversation_id == conversation_id)
      .where(Message.role == MessageRole.assistant)
      .order_by(Message.created_at.desc())
    ).first()
    if not msg:
      return False
    s.delete(msg)
    s.commit()
    return True


def get_last_user_message(conversation_id: UUID) -> Message | None:
  with get_session() as s:
    return s.exec(
      select(Message)
      .where(Message.conversation_id == conversation_id)
      .where(Message.role == MessageRole.user)
      .order_by(Message.created_at.desc())
    ).first()


def update_message_content(message_id: UUID, new_content: str) -> bool:
  with get_session() as s:
    msg = s.get(Message, message_id)
    if not msg:
      return False
    msg.content = new_content
    s.add(msg)
    s.commit()
    return True


def list_messages(conversation_id: UUID) -> list[Message]:
  with get_session() as s:
    return s.exec(
      select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at.asc())
    ).all()


def delete_conversation(conversation_id: UUID) -> bool:
  with get_session() as s:
    conv = s.get(Conversation, conversation_id)
    if not conv:
      return False
    # delete messages first (no cascade)
    msgs = s.exec(select(Message).where(Message.conversation_id == conversation_id)).all()
    for m in msgs:
      s.delete(m)
    s.delete(conv)
    s.commit()
    return True
