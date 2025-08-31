from __future__ import annotations

from base64 import b64encode
from logging import getLogger
from operator import itemgetter
from typing import TYPE_CHECKING, Any, TypedDict

from openai import OpenAI

from app.config import get_settings

logger = getLogger(__name__)

if TYPE_CHECKING:
  from collections.abc import Generator


class ModelInfo(TypedDict):
  id: str
  created: int | None
  owned_by: str | None


def get_client() -> OpenAI:
  s = get_settings()
  return OpenAI(api_key=s.OPENAI_API_KEY, base_url=s.OPENAI_BASE_URL)


def list_chat_models() -> list[ModelInfo]:
  client = get_client()
  models = client.models.list()
  out: list[ModelInfo] = [
    {
      'id': m.id,
      'created': getattr(m, 'created', None),
      'owned_by': getattr(m, 'owned_by', None),
    }
    for m in models
  ]
  out.sort(key=itemgetter('id'))
  return out


def _b64_data_url(image_bytes: bytes, mime: str) -> str:
  return f'data:{mime};base64,' + b64encode(image_bytes).decode('utf-8')


def build_user_content(
  text: str | None,
  images: list[tuple[bytes, str]] | None,
  extra_texts: list[str] | None = None,
) -> list[dict[str, Any]]:
  parts: list[dict[str, Any]] = []
  if text:
    parts.append({'type': 'text', 'text': text})
  if extra_texts:
    parts.extend({'type': 'text', 'text': t} for t in extra_texts if t)
  if images:
    for data, mime in images:
      parts.append({
        'type': 'image_url',
        'image_url': {'url': _b64_data_url(data, mime)},
      })
  return parts


def stream_chat_completion(
  model: str,
  messages: list[dict[str, Any]],
  temperature: float = 0.2,
  max_tokens: int | None = None,
) -> Generator[str]:
  client = get_client()
  stream = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=temperature,
    max_tokens=max_tokens,
    stream=True,
  )
  for chunk in stream:
    try:
      choices = getattr(chunk, 'choices', []) or []
      for ch in choices:
        delta = getattr(ch, 'delta', None)
        if delta is None:
          continue
        content = getattr(delta, 'content', None)
        if not content:
          continue
        if isinstance(content, str):
          yield content
        else:
          try:
            for part in content:
              text_val = getattr(part, 'text', None)
              if text_val:
                yield text_val
          except Exception:
            yield str(content)
    except Exception:
      logger.exception('Error processing streamed chat completion chunk')


def generate_title(model: str, first_user_message: str) -> str:
  client = get_client()
  system = 'You generate concise conversation titles, 6 words max, no punctuation.'
  prompt = 'Create a short title for this conversation in 6 words or fewer: ' + first_user_message
  resp = client.chat.completions.create(
    model=model,
    messages=[
      {'role': 'system', 'content': system},
      {'role': 'user', 'content': prompt},
    ],
    temperature=0.3,
    max_tokens=20,
  )
  title = resp.choices[0].message.content or 'Untitled'
  return title.strip().strip('\n').strip('"')[:80]
