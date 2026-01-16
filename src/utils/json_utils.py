import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


_SERVICE_TYPE_VALUES = {
    "service", "phone_call", "contact_registered", "chat_created",
    "user_joined", "user_left", "group_call", "pin_message",
}

_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]+",
    flags=re.UNICODE
)

_ALNUM_LETTER_RE = re.compile(r"[A-Za-zА-Яа-я0-9]", flags=re.UNICODE)

_MENTION_RE = re.compile(r"(?<!\w)@[\w_]{2,}", flags=re.UNICODE)

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _is_service_message(msg: Dict[str, Any]) -> bool:
    """
    Best-effort detection of Telegram service messages.
    Telegram exports vary by version; this tries common fields.
    """
    t = msg.get("type")
    if isinstance(t, str) and t.lower() in _SERVICE_TYPE_VALUES:
        return True

    action = msg.get("action")
    if isinstance(action, str) and action:
        return True

    return False


def _normalize_whitespace(s: str) -> str:
    return " ".join((s or "").split()).strip()


def _is_emoji_only(text: str) -> bool:
    """
    Returns True if text contains no letters/digits after removing spaces/punct.
    Emoji-only / sticker-like.
    """
    if not text:
        return True
    if _ALNUM_LETTER_RE.search(text):
        return False

    tmp = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    tmp = _EMOJI_RE.sub("", tmp)
    return len(tmp) == 0


def build_speaker_map(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map original 'from' names to Speaker_A, Speaker_B, ...
    Stable within one dataset build run.
    """
    speakers = []
    seen = set()
    for m in messages:
        fr = m.get("from")
        if isinstance(fr, str) and fr.strip():
            key = fr.strip()
            if key not in seen:
                seen.add(key)
                speakers.append(key)

    mapping = {}
    for i, sp in enumerate(speakers):
        mapping[sp] = f"Speaker_{chr(ord('A') + (i % 26))}" if i < 26 else f"Speaker_{i+1}"
    return mapping


def build_mention_map(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map @mentions to <user_1>, <user_2>, ...
    Uses both text_entities mentions and raw text @handles.
    """
    handles = []
    seen = set()

    for m in messages:
        for ent in m.get("text_entities") or []:
            if ent.get("type") == "mention":
                h = ent.get("text")
                if isinstance(h, str) and h.startswith("@"):
                    if h not in seen:
                        seen.add(h)
                        handles.append(h)

        text = m.get("text") or ""
        if isinstance(text, str):
            for h in _MENTION_RE.findall(text):
                if h not in seen:
                    seen.add(h)
                    handles.append(h)

    mapping = {}
    for i, h in enumerate(handles, start=1):
        mapping[h] = f"<user_{i}>"
    return mapping


def anonymize_text(text: str,
                   mention_map: Optional[Dict[str, str]] = None,
                   mask_urls: bool = True) -> str:
    s = text or ""
    s = _normalize_whitespace(s)

    if mask_urls:
        s = _URL_RE.sub("<url>", s)

    if mention_map:
        for h in sorted(mention_map.keys(), key=len, reverse=True):
            s = s.replace(h, mention_map[h])
    else:
        s = _MENTION_RE.sub("<user>", s)

    return s


def build_chat_name_map(chats: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Map original chat names -> Chat_001, Chat_002, ...
    Stable within one dataset build run.
    """
    names = []
    seen = set()
    for c in chats:
        name = c.get("name")
        if isinstance(name, str):
            name = name.strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)

    return {n: f"Chat_{i:03d}" for i, n in enumerate(names, start=1)}


def anonymize_chat_meta(chat: Dict[str, Any], chat_name_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Replace chat-level identifiers (currently only 'name').
    """
    out = dict(chat)
    name = out.get("name")
    if isinstance(name, str):
        key = name.strip()
        if chat_name_map and key in chat_name_map:
            out["name"] = chat_name_map[key]
        else:
            out["name"] = "Chat"
    return out


def anonymize_chat_messages(
    chat: Dict[str, Any],
    speaker_map: Optional[Dict[str, str]] = None,
    mention_map: Optional[Dict[str, str]] = None,
    mask_urls: bool = True,
    chat_name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Returns new chat dict where:
    - chat['name'] -> Chat_XXX (if map provided)
    - msg['from'] is replaced by stable Speaker_* id
    - mentions in text are masked (<user_n>)
    - optional url masking
    """
    chat = anonymize_chat_meta(chat, chat_name_map=chat_name_map)

    meta = {k: v for k, v in chat.items() if k != "messages"}
    msgs = chat.get("messages", []) or []

    if speaker_map is None:
        speaker_map = build_speaker_map(msgs)
    if mention_map is None:
        mention_map = build_mention_map(msgs)

    out_msgs = []
    for m in msgs:
        nm = dict(m)

        fr = nm.get("from")
        if isinstance(fr, str) and fr.strip():
            nm["from"] = speaker_map.get(fr.strip(), "Speaker_?")

        txt = nm.get("text") or ""
        if isinstance(txt, str):
            nm["text"] = anonymize_text(txt, mention_map=mention_map, mask_urls=mask_urls)

        out_msgs.append(nm)

    return {**meta, "messages": out_msgs}


def normalize_message_text(text):
    """
    Нормализует поле `text` из Telegram-JSON в плоскую строку.
    """
    if text is None:
        return ""

    if isinstance(text, str):
        return text.strip()

    if isinstance(text, list):
        parts: List[str] = []
        for item in text:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text", "") or ""
                href = item.get("href")
                if href and href != t:
                    parts.append(f"{t} ({href})")
                else:
                    parts.append(t)
        return "".join(parts).strip()

    return str(text).strip()


def normalize_chat_messages(
        chat: Dict[str, Any],
        drop_empty: bool = True,
        drop_service: bool = True,
        drop_emoji_only: bool = True,
        dedupe_consecutive: bool = True,
) -> Dict[str, Any]:
    """
      - normalize `text` into a string
      - optionally drop service messages
      - optionally drop empty messages
      - optionally drop emoji-only messages
      - optionally drop consecutive exact duplicates (same author + same text)
    """
    meta = {k: v for k, v in chat.items() if k != "messages"}
    normalized_messages: List[Dict[str, Any]] = []

    prev_key: Optional[Tuple[str, str]] = None

    for msg in chat.get("messages", []) or []:
        if drop_service and _is_service_message(msg):
            continue

        new_msg = dict(msg)
        new_msg["text"] = _normalize_whitespace(normalize_message_text(msg.get("text")))

        if drop_empty and not new_msg.get("text"):
            continue

        if drop_emoji_only and _is_emoji_only(new_msg["text"]):
            continue

        if dedupe_consecutive:
            author = (new_msg.get("from") or "").strip()
            key = (author, new_msg["text"])
            if prev_key == key:
                continue
            prev_key = key

        normalized_messages.append(new_msg)

    return {**meta, "messages": normalized_messages}


def split_chat_into_batches(
        chat: Dict[str, Any],
        batch_size: int = 20) -> List[Dict[str, Any]]:
    """
    Делит чат на батчи по batch_size сообщений.
    Оставляет все нормализованные сообщения как есть.
    """
    messages = chat.get("messages", [])
    meta = {k: v for k, v in chat.items() if k != "messages"}

    batches: List[Dict[str, Any]] = []
    for i in range(0, len(messages), batch_size):
        batch_messages = messages[i:i + batch_size]
        batch = {**meta, "messages": batch_messages}
        batches.append(batch)

    return batches


def batch_to_dialog_text(batch: Dict[str, Any], max_chars: int = 5000) -> str:
    """
    Превращает батч в плоский текст диалога вида:
    2025-07-09T11:59:27 | Автор: текст
    и обрезает до max_chars.
    """
    lines = []
    for msg in batch.get("messages", []):
        date = msg.get("date", "") or ""
        author = msg.get("from", "") or "❓"
        text = msg.get("text", "") or ""
        text_one_line = " ".join(text.split())
        if not text_one_line:
            continue
        lines.append(f"{date} | {author}: {text_one_line}")

    full = "\n".join(lines)
    if len(full) > max_chars:
        return full[:max_chars] + " ..."
    return full
