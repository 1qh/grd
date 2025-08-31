from __future__ import annotations

from collections.abc import Generator  # noqa: TC003
from contextlib import suppress
from logging import getLogger
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from uuid import UUID

from dotenv import load_dotenv
from gradio import (
  HTML,
  Blocks,
  Button,
  Chatbot,
  ChatMessage,
  Column,
  Dropdown,
  EditData,
  Group,
  Markdown,
  MessageDict,
  MultimodalTextbox,
  Request,
  RetryData,
  Row,
  State,
  Textbox,
  update,
)

from app.config import get_settings
from app.db import init_db
from app.llm import build_user_content, generate_title, list_chat_models, stream_chat_completion
from app.models import MessageRole
from app.service import (
  add_message,
  create_conversation,
  delete_conversation,
  delete_last_assistant_message,
  get_conversation,
  get_last_user_message,
  list_conversations,
  list_messages,
  rename_conversation,
  update_message_content,
)

load_dotenv(override=False)
_ = get_settings()
init_db()

logger = getLogger(__name__)


UPLOAD_TEXT_LIMIT = 200_000


def _is_image_file(path: str) -> bool:
  mime, _ = guess_type(path)
  return bool(mime and mime.startswith('image/'))


def _is_text_file(path: str) -> bool:
  mime, _ = guess_type(path)
  if not mime:
    return Path(path).suffix.lower() in {
      '.py',
      '.txt',
      '.md',
      '.json',
      '.csv',
      '.yaml',
      '.yml',
      '.toml',
      '.js',
      '.ts',
      '.tsx',
      '.jsx',
      '.java',
      '.go',
      '.rs',
      '.c',
      '.cpp',
      '.h',
      '.hpp',
      '.sh',
    }
  return mime.startswith('text/') or mime in {'application/json', 'application/xml'}


def _read_text_file(path: str) -> str:
  try:
    with Path(path).open(encoding='utf-8', errors='ignore') as f:
      data = f.read(UPLOAD_TEXT_LIMIT + 1)
      if len(data) > UPLOAD_TEXT_LIMIT:
        data = data[:UPLOAD_TEXT_LIMIT] + '\n...[truncated]'
      return data
  except Exception:
    logger.exception('Failed to read text file: %s', path)
    return ''


def _build_sidebar_html(active: str | None = None) -> str:
  items = list_conversations(limit=200)
  if not items:
    return "<div class='sidebar-actions'><a class='new-chat' href='/'>+ New Chat</a></div><div class='muted'>No conversations yet.</div>"
  lines: list[str] = [
    "<div class='sidebar-actions'><a class='new-chat' href='/'>+ New Chat</a></div>",
    "<div class='conv-list'>",
  ]
  for c in items:
    title = (c.title or 'Untitled').replace('<', '&lt;').replace('>', '&gt;')
    is_active = ' active' if active and str(c.id) == str(active) else ''
    lines.append(
      "<div class='conv-item'>"
      f"<a class='conv-link{is_active}' href='/?cid={c.id}'>{title}</a>"
      f"<a class='icon' title='Rename' href='/?cid={c.id}&rename=1'>✏️</a>"
      f"<a class='icon danger' title='Delete' href='/?cid={c.id}&delete=1' onclick=\"return confirm('Delete this conversation?')\">🗑️</a>"
      '</div>'
    )
  lines.append('</div>')
  return '\n'.join(lines)


def _load_on_start(
  request: Request, model_dropdown: str | None
) -> tuple[str, list[ChatMessage], str, dict[str, Any], dict[str, Any], dict[str, Any], str]:
  params = dict(getattr(request, 'query_params', {}) or {})
  cid_param = params.get('cid') or params.get('conversation_id')
  model_list = list_chat_models()
  model_ids = [m['id'] for m in model_list]
  selected_model = model_dropdown or (model_ids[0] if model_ids else '')
  if cid_param:
    try:
      cid = UUID(str(cid_param))
    except Exception:
      cid = None
  else:
    cid = None
  chat_messages: list[ChatMessage] = []
  want_rename = bool(params.get('rename'))
  want_delete = bool(params.get('delete'))
  if cid:
    conv = get_conversation(cid)
    if conv:
      for m in list_messages(conv.id):
        role = 'user' if m.role == MessageRole.user else 'assistant'
        if m.role == MessageRole.system:
          continue
        chat_messages.append(ChatMessage(content=m.content, role=role))
      if conv.model:
        selected_model = conv.model
      if want_delete:
        with suppress(Exception):
          delete_conversation(conv.id)
        return (
          '',
          [],
          _build_sidebar_html(None),
          update(choices=model_ids, value=selected_model),
          update(visible=False),
          update(value=''),
          '',
        )
      if want_rename:
        return (
          str(conv.id),
          chat_messages,
          _build_sidebar_html(str(conv.id)),
          update(choices=model_ids, value=selected_model),
          update(visible=True),
          update(value=(conv.title or 'Untitled')),
          str(conv.id),
        )
      return (
        str(conv.id),
        chat_messages,
        _build_sidebar_html(str(conv.id)),
        update(choices=model_ids, value=selected_model),
        update(visible=False),
        update(value=''),
        str(conv.id),
      )
  return (
    '',
    chat_messages,
    _build_sidebar_html(None),
    update(choices=model_ids, value=selected_model),
    update(visible=False),
    update(value=''),
    '',
  )


def _fetch_models() -> list[str]:
  return [m['id'] for m in list_chat_models()]


def _messages_for_llm(
  conversation_id: UUID,
  current_user_text: str | None,
  current_images: list[tuple[bytes, str]] | None,
  current_extra_texts: list[str] | None,
) -> list[dict[str, Any]]:
  msgs: list[dict[str, Any]] = []
  history = list_messages(conversation_id)
  for m in history:
    if m.role == MessageRole.system:
      msgs.append({'role': 'system', 'content': m.content})
    elif m.role == MessageRole.user:
      msgs.append({'role': 'user', 'content': [{'type': 'text', 'text': m.content}]})
    elif m.role == MessageRole.assistant:
      msgs.append({'role': 'assistant', 'content': m.content})
  parts = build_user_content(current_user_text or '', current_images or [], current_extra_texts or [])
  msgs.append({'role': 'user', 'content': parts})
  return msgs


def _prepare_current_parts(files: list[str] | None) -> tuple[list[str], list[tuple[bytes, str]]]:
  extra_texts: list[str] = []
  images: list[tuple[bytes, str]] = []
  for p in files or []:
    if _is_text_file(p):
      extra_texts.append(_read_text_file(p))
    elif _is_image_file(p):
      mime, _ = guess_type(p)
      try:
        images.append((Path(p).read_bytes(), mime or 'image/png'))
      except Exception:
        logger.exception('Failed to read image file: %s', p)
    else:
      try:
        size = Path(p).stat().st_size
        if size <= 512 * 1024:
          data = Path(p).read_bytes()
          extra_texts.append(data.decode('utf-8', errors='ignore'))
      except Exception:
        logger.exception('Failed to read attachment file: %s', p)
  return extra_texts, images


def _create_conv_if_needed(current_cid: str, model: str) -> UUID:
  if current_cid:
    return UUID(current_cid)
  conv = create_conversation(model=model)
  return conv.id


def on_send(
  data: dict[str, Any] | None,
  cid_state: str,
  model: str,
  _request: Request,
) -> Generator[tuple[list[ChatMessage], str, str, str]]:
  if not data:
    yield [], '', cid_state, _build_sidebar_html(cid_state or None)
    return
  text: str = (data.get('text') or '').strip()
  files: list[str] = data.get('files') or []

  conv_id = _create_conv_if_needed(cid_state, model)
  add_message(conversation_id=conv_id, role=MessageRole.user, content=text, attachments=[{'path': p} for p in files])

  messages_ui: list[ChatMessage] = []
  conv = get_conversation(conv_id)
  if conv:
    for m in list_messages(conv.id):
      if m.role == MessageRole.user:
        messages_ui.append(ChatMessage(role='user', content=m.content))
      elif m.role == MessageRole.assistant:
        messages_ui.append(ChatMessage(role='assistant', content=m.content))

  messages_ui.append(ChatMessage(role='assistant', content=''))
  yield messages_ui, '', str(conv_id), _build_sidebar_html(str(conv_id))

  extra_texts, images = _prepare_current_parts(files)
  llm_messages = _messages_for_llm(conv_id, text, images, extra_texts)

  buffer = []
  for token in stream_chat_completion(model=model, messages=llm_messages):
    buffer.append(token)
    messages_ui[-1] = ChatMessage(role='assistant', content=''.join(buffer))
    yield messages_ui, '', str(conv_id), _build_sidebar_html(str(conv_id))

  full_assistant = ''.join(buffer)
  add_message(conversation_id=conv_id, role=MessageRole.assistant, content=full_assistant)

  conv = get_conversation(conv_id)
  if conv and not conv.title:
    try:
      first_user = next((m for m in list_messages(conv.id) if m.role == MessageRole.user), None)
      if first_user and first_user.content:
        title = generate_title(model=model, first_user_message=first_user.content)
        rename_conversation(conv.id, title)
    except Exception:
      logger.exception('Failed to auto-generate conversation title')
  yield messages_ui, '', str(conv_id), _build_sidebar_html(str(conv_id))


def on_retry(cid_state: str, model: str) -> Generator[tuple[list[ChatMessage], str, str, str]]:
  if not cid_state:
    yield [], '', cid_state, _build_sidebar_html(None)
    return
  conv_id = UUID(cid_state)
  last_user = get_last_user_message(conv_id)
  if not last_user:
    yield [], '', cid_state, _build_sidebar_html(cid_state)
    return
  delete_last_assistant_message(conv_id)
  conv = get_conversation(conv_id)
  messages_ui: list[ChatMessage] = []
  if conv:
    for m in list_messages(conv.id):
      if m.role == MessageRole.user:
        messages_ui.append(ChatMessage(role='user', content=m.content))
      elif m.role == MessageRole.assistant:
        messages_ui.append(ChatMessage(role='assistant', content=m.content))
  messages_ui.append(ChatMessage(role='assistant', content=''))
  yield messages_ui, '', cid_state, _build_sidebar_html(cid_state)

  extra_texts, images = ([], [])
  llm_messages = _messages_for_llm(conv_id, last_user.content, images, extra_texts)
  buffer: list[str] = []
  for token in stream_chat_completion(model=model, messages=llm_messages):
    buffer.append(token)
    messages_ui[-1] = ChatMessage(role='assistant', content=''.join(buffer))
    yield messages_ui, '', cid_state, _build_sidebar_html(cid_state)
  full = ''.join(buffer)
  add_message(conversation_id=conv_id, role=MessageRole.assistant, content=full)
  yield messages_ui, '', cid_state, _build_sidebar_html(cid_state)


def on_start_edit(cid_state: str) -> tuple[str, dict[str, Any]]:
  if not cid_state:
    return '', update(visible=False)
  conv_id = UUID(cid_state)
  last_user = get_last_user_message(conv_id)
  if not last_user:
    return '', update(visible=False)
  return last_user.content, update(visible=True)


def on_apply_edit(
  cid_state: str, new_text: str, model: str
) -> Generator[tuple[list[ChatMessage], str, str, str, dict[str, Any], dict[str, Any]]]:
  if not cid_state:
    yield [], '', cid_state, _build_sidebar_html(None), update(visible=False), update(value='')
    return
  conv_id = UUID(cid_state)
  last_user = get_last_user_message(conv_id)
  if not last_user:
    yield [], '', cid_state, _build_sidebar_html(cid_state), update(visible=False), update(value='')
    return
  update_message_content(last_user.id, new_text)
  delete_last_assistant_message(conv_id)

  conv = get_conversation(conv_id)
  messages_ui: list[ChatMessage] = []
  if conv:
    for m in list_messages(conv.id):
      if m.role == MessageRole.user:
        messages_ui.append(ChatMessage(role='user', content=m.content))
      elif m.role == MessageRole.assistant:
        messages_ui.append(ChatMessage(role='assistant', content=m.content))
  messages_ui.append(ChatMessage(role='assistant', content=''))
  yield messages_ui, '', cid_state, _build_sidebar_html(cid_state), update(visible=False), update(value='')

  llm_messages = _messages_for_llm(conv_id, new_text, [], [])
  buffer: list[str] = []
  for token in stream_chat_completion(model=model, messages=llm_messages):
    buffer.append(token)
    messages_ui[-1] = ChatMessage(role='assistant', content=''.join(buffer))
    yield messages_ui, '', cid_state, _build_sidebar_html(cid_state), update(visible=False), update(value='')
  full = ''.join(buffer)
  add_message(conversation_id=conv_id, role=MessageRole.assistant, content=full)
  yield messages_ui, '', cid_state, _build_sidebar_html(cid_state), update(visible=False), update(value='')


def on_rename(cid_state: str, new_title: str) -> str:
  if not cid_state:
    return new_title
  with suppress(Exception):
    rename_conversation(UUID(cid_state), new_title)
  return new_title


def on_refresh_models() -> dict[str, Any]:
  return update(choices=_fetch_models())


def build_app() -> Blocks:
  with Blocks(css_paths='app/static/styles.css') as demo:
    with Row():
      # Fixed sidebar (non-collapsible)
      with Column(scale=0, min_width=300, elem_id='fixed-sidebar'):
        model_dropdown = Dropdown(choices=[], value=None, label=None, allow_custom_value=True, show_label=False)
        refresh_btn = Button('Reload models', size='sm')
        sidebar_html = HTML(value=_build_sidebar_html(None))
      with Column(scale=4):
        cid_state = State(value='')
        # Rename modal overlay
        rename_target = State(value='')
        with Group(visible=False, elem_id='rename-modal') as rename_modal:
          Markdown('### Rename Conversation')
          rename_tb = Textbox(value='', label=None, placeholder='Enter a new title', show_label=False)
          with Row():
            rename_save = Button('Save', variant='primary', size='sm')
            rename_cancel = Button('Cancel', variant='secondary', size='sm')
        chat = Chatbot(
          height=600,
          type='messages',
          render_markdown=True,
          layout='bubble',
          show_label=False,
          editable='user',
          show_copy_button=True,
          feedback_options=None,
        )
        input_box = MultimodalTextbox(
          label=None,
          placeholder='Send a message or upload files...',
          file_count='multiple',
          file_types=['.txt', '.md', '.py', '.json', '.png', '.jpg', '.jpeg', '.webp', '.gif'],
          submit_btn=True,
          show_label=False,
        )

    demo.load(
      _load_on_start,
      inputs=[model_dropdown],
      outputs=[cid_state, chat, sidebar_html, model_dropdown, rename_modal, rename_tb, rename_target],
      queue=False,
    )

    input_box.submit(
      on_send,
      inputs=[input_box, cid_state, model_dropdown],
      outputs=[chat, input_box, cid_state, sidebar_html],
    )

    # Rename modal actions
    def on_rename_save(target_id: str, new_title: str) -> tuple[dict[str, Any], str, str]:
      if not target_id:
        return update(visible=False), _build_sidebar_html(None), ''
      with suppress(Exception):
        rename_conversation(UUID(target_id), new_title)
      return update(visible=False), _build_sidebar_html(target_id), ''

    rename_save.click(
      on_rename_save, inputs=[rename_target, rename_tb], outputs=[rename_modal, sidebar_html, rename_target]
    )
    rename_cancel.click(lambda: update(visible=False), inputs=[], outputs=[rename_modal], queue=False)

    def _is_last_user_index(history: list[MessageDict], idx: int) -> bool:
      last_idx = -1
      for i in range(len(history) - 1, -1, -1):
        if history[i]['role'] == 'user':
          last_idx = i
          break
      return idx == last_idx

    def handle_retry_event(
      retry_data: RetryData, history: list[MessageDict], cid_state_val: str, model_val: str
    ) -> Generator[tuple[list[ChatMessage], str, str, str]]:
      if not cid_state_val:
        yield history, '', cid_state_val, _build_sidebar_html(None)
        return
      idx = retry_data.index[0] if isinstance(retry_data.index, tuple) else retry_data.index
      if not _is_last_user_index(history, int(idx)):
        yield history, '', cid_state_val, _build_sidebar_html(cid_state_val)
        return
      conv_id = UUID(cid_state_val)
      last_user = get_last_user_message(conv_id)
      if not last_user:
        yield history, '', cid_state_val, _build_sidebar_html(cid_state_val)
        return
      delete_last_assistant_message(conv_id)
      messages_ui: list[ChatMessage] = []
      for m in list_messages(conv_id):
        if m.role == MessageRole.user:
          messages_ui.append(ChatMessage(role='user', content=m.content))
        elif m.role == MessageRole.assistant:
          messages_ui.append(ChatMessage(role='assistant', content=m.content))
      messages_ui.append(ChatMessage(role='assistant', content=''))
      yield messages_ui, '', cid_state_val, _build_sidebar_html(cid_state_val)
      buffer: list[str] = []
      for token in stream_chat_completion(
        model=model_val, messages=_messages_for_llm(conv_id, last_user.content, [], [])
      ):
        buffer.append(token)
        messages_ui[-1] = ChatMessage(role='assistant', content=''.join(buffer))
        yield messages_ui, '', cid_state_val, _build_sidebar_html(cid_state_val)
      add_message(conversation_id=conv_id, role=MessageRole.assistant, content=''.join(buffer))
      yield messages_ui, '', cid_state_val, _build_sidebar_html(cid_state_val)

    chat.retry(
      handle_retry_event, inputs=[chat, cid_state, model_dropdown], outputs=[chat, input_box, cid_state, sidebar_html]
    )

    def handle_edit_event(
      edit_data: EditData, history: list[MessageDict], cid_state_val: str, model_val: str
    ) -> Generator[tuple[list[ChatMessage], str, str, str]]:
      if not cid_state_val:
        yield history, '', cid_state_val, _build_sidebar_html(None)
        return
      idx = edit_data.index[0] if isinstance(edit_data.index, tuple) else edit_data.index
      if not _is_last_user_index(history, int(idx)):
        yield history, '', cid_state_val, _build_sidebar_html(cid_state_val)
        return
      conv_id = UUID(cid_state_val)
      last_user = get_last_user_message(conv_id)
      if not last_user:
        yield history, '', cid_state_val, _build_sidebar_html(cid_state_val)
        return
      update_message_content(last_user.id, edit_data.value)
      delete_last_assistant_message(conv_id)
      messages_ui: list[ChatMessage] = []
      for m in list_messages(conv_id):
        if m.role == MessageRole.user:
          messages_ui.append(ChatMessage(role='user', content=m.content))
        elif m.role == MessageRole.assistant:
          messages_ui.append(ChatMessage(role='assistant', content=m.content))
      messages_ui.append(ChatMessage(role='assistant', content=''))
      yield messages_ui, '', cid_state_val, _build_sidebar_html(cid_state_val)
      buffer: list[str] = []
      for token in stream_chat_completion(
        model=model_val, messages=_messages_for_llm(conv_id, edit_data.value, [], [])
      ):
        buffer.append(token)
        messages_ui[-1] = ChatMessage(role='assistant', content=''.join(buffer))
        yield messages_ui, '', cid_state_val, _build_sidebar_html(cid_state_val)
      add_message(conversation_id=conv_id, role=MessageRole.assistant, content=''.join(buffer))
      yield messages_ui, '', cid_state_val, _build_sidebar_html(cid_state_val)

    chat.edit(
      handle_edit_event, inputs=[chat, cid_state, model_dropdown], outputs=[chat, input_box, cid_state, sidebar_html]
    )

    refresh_btn.click(on_refresh_models, inputs=[], outputs=[model_dropdown], queue=False)

  return demo


if __name__ == '__main__':
  app = build_app()
  app.launch()
