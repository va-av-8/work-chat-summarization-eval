from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import numpy as np

from src.eval.embedding_backend import EmbeddingBackend


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-]+", flags=re.UNICODE)


def norm_item(s: str) -> str:
    s = (s or "").strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def prf1_set(ref_items: List[str], gen_items: List[str]) -> Dict[str, float]:
    ref = {norm_item(x) for x in (ref_items or []) if norm_item(x)}
    gen = {norm_item(x) for x in (gen_items or []) if norm_item(x)}

    if not ref and not gen:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

    tp = len(ref & gen)
    fp = len(gen - ref)
    fn = len(ref - gen)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def micro_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def context_similarity(embedder: EmbeddingBackend, ref_context: str, gen_context: str) -> float:
    ref_context = (ref_context or "").strip()
    gen_context = (gen_context or "").strip()

    if not ref_context and not gen_context:
        return 1.0
    if not ref_context or not gen_context:
        return 0.0

    v = embedder.embed([ref_context, gen_context])
    # normalized => cosine == dot
    return float(np.dot(v[0], v[1]))


def action_desc(a: Dict[str, Any]) -> str:
    return (a.get("description") or "").strip()


def _normalize_slot(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


@dataclass
class ActionMatchResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    assignee_acc: float
    deadline_acc: float
    matches: List[Dict[str, Any]]  # each: {sim, ref_i, gen_j}


def match_actions_by_description(
    embedder: EmbeddingBackend,
    ref_actions: List[Dict[str, Any]],
    gen_actions: List[Dict[str, Any]],
    threshold: float,
) -> ActionMatchResult:
    ref_desc = [action_desc(a) for a in (ref_actions or []) if action_desc(a)]
    gen_desc = [action_desc(a) for a in (gen_actions or []) if action_desc(a)]

    if not ref_desc and not gen_desc:
        return ActionMatchResult(1.0, 1.0, 1.0, 0, 0, 0, 1.0, 1.0, [])

    if not ref_desc or not gen_desc:
        tp = 0
        fp = len(gen_desc)
        fn = len(ref_desc)
        return ActionMatchResult(0.0, 0.0, 0.0, tp, fp, fn, 0.0, 0.0, [])

    ref_vecs = embedder.embed(ref_desc)
    gen_vecs = embedder.embed(gen_desc)

    sims = ref_vecs @ gen_vecs.T  # (R, G) cosine since normalized

    pairs: List[Tuple[float, int, int]] = []
    R, G = sims.shape
    for i in range(R):
        for j in range(G):
            pairs.append((float(sims[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    used_r, used_g = set(), set()
    matches = []
    for sim, i, j in pairs:
        if sim < threshold:
            break
        if i in used_r or j in used_g:
            continue
        used_r.add(i)
        used_g.add(j)
        matches.append({"sim": sim, "ref_i": i, "gen_j": j})

    tp = len(matches)
    fp = len(gen_desc) - tp
    fn = len(ref_desc) - tp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # slot accuracy on matched pairs
    # map desc-index -> first action dict with that desc
    ref_map = []
    for d in ref_desc:
        ref_map.append(next(a for a in ref_actions if action_desc(a) == d))
    gen_map = []
    for d in gen_desc:
        gen_map.append(next(a for a in gen_actions if action_desc(a) == d))

    ass_ok = 0
    ddl_ok = 0
    for m in matches:
        ra = ref_map[m["ref_i"]]
        ga = gen_map[m["gen_j"]]
        if _normalize_slot(ra.get("assignee")) == _normalize_slot(ga.get("assignee")):
            ass_ok += 1
        if _normalize_slot(ra.get("deadline")) == _normalize_slot(ga.get("deadline")):
            ddl_ok += 1

    assignee_acc = ass_ok / tp if tp else 0.0
    deadline_acc = ddl_ok / tp if tp else 0.0

    return ActionMatchResult(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        assignee_acc=assignee_acc,
        deadline_acc=deadline_acc,
        matches=matches,
    )


def overall_auto_score(context_sim: float, act_f1: float, dec_f1: float, q_f1: float) -> float:
    # вспомогательная, не делаем её "главной"
    return 0.35 * context_sim + 0.35 * act_f1 + 0.15 * dec_f1 + 0.15 * q_f1
