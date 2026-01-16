from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

SUMMARY_KEYS = {"Context", "Decisions", "Actions", "Questions"}


def _try_load_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = (text or "").strip()
    if not text:
        return None, "empty response"

    try:
        return json.loads(text), None
    except Exception as e:
        last_err = f"direct json.loads failed: {e}"

    m = _CODE_BLOCK_RE.search(text)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate), None
        except Exception as e:
            last_err = f"code block json.loads failed: {e}"

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last + 1]
        try:
            return json.loads(candidate), None
        except Exception as e:
            last_err = f"brace slice json.loads failed: {e}"

    return None, last_err


def _find_summary_block(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        if SUMMARY_KEYS.intersection(obj.keys()):
            return obj
        for v in obj.values():
            found = _find_summary_block(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_summary_block(item)
            if found is not None:
                return found
    return None


def _normalize_summary_block(block: Dict[str, Any]) -> Dict[str, Any]:
    context = block.get("Context") or ""
    decisions = block.get("Decisions") or []
    actions = block.get("Actions") or []
    questions = block.get("Questions") or []

    if not isinstance(decisions, list):
        decisions = [decisions]
    decisions = [str(d) for d in decisions if d is not None]

    if not isinstance(questions, list):
        questions = [questions]
    questions = [str(q) for q in questions if q is not None]

    norm_actions: List[Dict[str, Any]] = []
    if isinstance(actions, dict):
        actions = [actions]
    if isinstance(actions, list):
        for a in actions:
            if isinstance(a, dict):
                norm_actions.append({
                    "assignee": a.get("assignee", None),
                    "description": str(a.get("description", "") or ""),
                    "deadline": a.get("deadline", None),
                })
            else:
                norm_actions.append({"assignee": None, "description": str(a), "deadline": None})

    return {
        "Context": str(context),
        "Decisions": decisions,
        "Actions": norm_actions,
        "Questions": questions,
    }


def parse_structured_summary(raw_text: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Возвращает (summary_dict, parse_error).
    summary_dict всегда содержит 4 ключа (Context/Decisions/Actions/Questions).
    """
    data, err = _try_load_json(raw_text)
    if data is None:
        empty = {"Context": "", "Decisions": [], "Actions": [], "Questions": []}
        return empty, err or "unknown json parse error"

    block = _find_summary_block(data)
    if block is None:
        empty = {"Context": "", "Decisions": [], "Actions": [], "Questions": []}
        return empty, "no summary-like keys found"

    return _normalize_summary_block(block), None
