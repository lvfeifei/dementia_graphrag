"""
Step 07: Session memory for multi-turn dementia care GraphRAG agent.

This module manages short-term local conversation memory.

Responsibilities:
1. Create session
2. Read session
3. Append user / assistant turns
4. Update active_problem_ids and active_problems
5. Save care recipient profile
6. Generate conversation summary

This script does NOT call DeepSeek.
This script does NOT read or modify Neo4j.
This script does NOT perform retrieval.
It only reads/writes local JSON files under outputs/sessions/.

Recommended use:
- Step 08 agent will call this module to make the system multi-turn.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SESSIONS_DIR = PROJECT_ROOT / "outputs" / "sessions"
ANSWERS_DIR = PROJECT_ROOT / "outputs" / "answers"
RETRIEVAL_DEBUG_DIR = PROJECT_ROOT / "outputs" / "retrieval_debug"

DEFAULT_SESSION_ID = "demo_001"

LAST_ANSWER_JSON_PATH = ANSWERS_DIR / "last_answer_result.json"
LAST_ANSWER_MD_PATH = ANSWERS_DIR / "last_answer.md"
LAST_RETRIEVAL_RESULT_PATH = RETRIEVAL_DEBUG_DIR / "last_retrieval_result.json"

MAX_TURNS_KEEP = 20
MAX_RECENT_TURNS_IN_SUMMARY = 8


RISK_PROBLEM_IDS = {
    "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
    "H2", "H13", "H14",
}

RISK_KEYWORDS = [
    "走失", "外跑", "跌倒", "坠床", "攻击", "打人", "咬人", "踢人",
    "自伤", "伤人", "误服", "误吸", "噎食", "呛咳", "窒息",
    "幻觉", "妄想", "被害", "偷", "有人害",
]


RELATION_PATTERNS = [
    ("奶奶", "奶奶"),
    ("爷爷", "爷爷"),
    ("妈妈", "妈妈"),
    ("母亲", "妈妈"),
    ("妈", "妈妈"),
    ("爸爸", "爸爸"),
    ("父亲", "爸爸"),
    ("爸", "爸爸"),
    ("外婆", "外婆"),
    ("外公", "外公"),
    ("姥姥", "姥姥"),
    ("姥爷", "姥爷"),
    ("老人", "老人"),
    ("长辈", "长辈"),
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_dirs() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def safe_session_id(session_id: str) -> str:
    session_id = clean_text(session_id) or DEFAULT_SESSION_ID
    session_id = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", session_id)
    return session_id[:80] or DEFAULT_SESSION_ID


def session_path(session_id: str) -> Path:
    ensure_dirs()
    sid = safe_session_id(session_id)
    return SESSIONS_DIR / f"{sid}.json"


def relative_path(path: Optional[Path]) -> str:
    if not path:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"JSON file should contain an object: {path}")

    return data


def read_markdown_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def make_empty_session(session_id: str) -> Dict[str, Any]:
    sid = safe_session_id(session_id)

    return {
        "session_id": sid,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "project": "dementia_care_graphrag_workflow_v01",
        "kg_version": "v02_final",

        "care_recipient_profile": {
            "display_name": "",
            "relation": "",
            "age_or_stage": "",
            "living_context": "",
            "main_caregiver": "",
            "notes": [],
            "recent_user_descriptions": [],
        },

        "active_problem_ids": [],
        "active_problems": [],

        "risk_flags": [],

        "conversation_turns": [],

        "conversation_summary": "",

        "last_user_question": "",
        "last_answer_summary": "",
        "last_retrieval_result_path": "",
        "last_answer_path": "",
        "last_answer_result_path": "",

        "metadata": {
            "memory_type": "short_term_local_json",
            "max_turns_keep": MAX_TURNS_KEEP,
            "note": "This is local short-term session memory for multi-turn care support. It should assist context understanding but must not replace KG retrieval.",
        },
    }


def create_session(session_id: str, overwrite: bool = False) -> Dict[str, Any]:
    ensure_dirs()

    path = session_path(session_id)

    if path.exists() and not overwrite:
        return load_session(session_id)

    if path.exists() and overwrite:
        backup_path = path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(path, backup_path)

    session = make_empty_session(session_id)
    save_session(session)

    return session


def load_session(session_id: str, create_if_missing: bool = True) -> Dict[str, Any]:
    ensure_dirs()

    path = session_path(session_id)

    if not path.exists():
        if create_if_missing:
            return create_session(session_id=session_id, overwrite=False)
        raise FileNotFoundError(f"Session not found: {path}")

    data = load_json_file(path)

    # Minimal schema migration / safety fill.
    empty = make_empty_session(session_id)
    for key, value in empty.items():
        if key not in data:
            data[key] = value

    for key, value in empty["care_recipient_profile"].items():
        data.setdefault("care_recipient_profile", {})
        if key not in data["care_recipient_profile"]:
            data["care_recipient_profile"][key] = value

    return data


def save_session(session: Dict[str, Any]) -> Path:
    ensure_dirs()

    sid = safe_session_id(session.get("session_id", DEFAULT_SESSION_ID))
    session["session_id"] = sid
    session["updated_at"] = now_iso()

    path = session_path(sid)

    path.write_text(
        json.dumps(session, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


def infer_profile_from_text(text: str) -> Dict[str, Any]:
    """
    Very light rule-based profile inference.
    It does not diagnose anything.
    It only tries to infer relation/display_name from user wording.
    """
    text = clean_text(text)

    result: Dict[str, Any] = {}

    for pattern, relation in RELATION_PATTERNS:
        if pattern in text:
            result["relation"] = relation
            result["display_name"] = relation
            break

    return result


def update_care_recipient_profile(
    session: Dict[str, Any],
    profile_updates: Optional[Dict[str, Any]] = None,
    source_text: str = "",
) -> Dict[str, Any]:
    profile = session.setdefault("care_recipient_profile", {})

    inferred = infer_profile_from_text(source_text) if source_text else {}

    updates: Dict[str, Any] = {}
    updates.update(inferred)
    updates.update(profile_updates or {})

    for key, value in updates.items():
        if value is None:
            continue

        if key == "notes":
            if not isinstance(value, list):
                value = [value]
            profile.setdefault("notes", [])
            for note in value:
                note = clean_text(note)
                if note and note not in profile["notes"]:
                    profile["notes"].append(note)
            continue

        if key == "recent_user_descriptions":
            if not isinstance(value, list):
                value = [value]
            profile.setdefault("recent_user_descriptions", [])
            for desc in value:
                desc = clean_text(desc)
                if desc and desc not in profile["recent_user_descriptions"]:
                    profile["recent_user_descriptions"].append(desc)
            profile["recent_user_descriptions"] = profile["recent_user_descriptions"][-8:]
            continue

        value_text = clean_text(value)
        if value_text:
            # Do not overwrite an existing display name/relation unless the new value is more specific.
            old = clean_text(profile.get(key))
            if not old or old == "老人":
                profile[key] = value_text

    if source_text:
        profile.setdefault("recent_user_descriptions", [])
        short = clean_text(source_text)
        if short and short not in profile["recent_user_descriptions"]:
            profile["recent_user_descriptions"].append(short)
        profile["recent_user_descriptions"] = profile["recent_user_descriptions"][-8:]

    session["care_recipient_profile"] = profile
    return session


def make_turn(
    role: str,
    content: str,
    summary: str = "",
    problem_ids: Optional[List[str]] = None,
    problem_names: Optional[List[str]] = None,
    retrieval_result_path: str = "",
    answer_path: str = "",
    answer_result_path: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    role = clean_text(role).lower()
    if role not in {"user", "assistant", "system"}:
        raise ValueError("role must be user, assistant, or system.")

    return {
        "turn_id": str(uuid.uuid4())[:12],
        "created_at": now_iso(),
        "role": role,
        "content": clean_text(content),
        "summary": clean_text(summary),
        "problem_ids": normalize_problem_ids(problem_ids),
        "problem_names": [clean_text(x) for x in (problem_names or []) if clean_text(x)],
        "retrieval_result_path": clean_text(retrieval_result_path),
        "answer_path": clean_text(answer_path),
        "answer_result_path": clean_text(answer_result_path),
        "extra": extra or {},
    }


def normalize_problem_ids(problem_ids: Optional[List[str]]) -> List[str]:
    result: List[str] = []
    seen = set()

    for pid in problem_ids or []:
        pid = clean_text(pid).upper()
        if not pid:
            continue
        if pid not in seen:
            result.append(pid)
            seen.add(pid)

    return result


def append_turn(
    session: Dict[str, Any],
    role: str,
    content: str,
    summary: str = "",
    problem_ids: Optional[List[str]] = None,
    problem_names: Optional[List[str]] = None,
    retrieval_result_path: str = "",
    answer_path: str = "",
    answer_result_path: str = "",
    extra: Optional[Dict[str, Any]] = None,
    avoid_duplicate_latest: bool = True,
) -> Dict[str, Any]:
    turns = session.setdefault("conversation_turns", [])

    content_clean = clean_text(content)

    if avoid_duplicate_latest and turns:
        latest = turns[-1]
        if latest.get("role") == role and clean_text(latest.get("content")) == content_clean:
            return session

    turn = make_turn(
        role=role,
        content=content_clean,
        summary=summary,
        problem_ids=problem_ids,
        problem_names=problem_names,
        retrieval_result_path=retrieval_result_path,
        answer_path=answer_path,
        answer_result_path=answer_result_path,
        extra=extra,
    )

    turns.append(turn)
    session["conversation_turns"] = turns[-MAX_TURNS_KEEP:]

    if role == "user":
        session["last_user_question"] = content_clean
        update_care_recipient_profile(session, source_text=content_clean)

    if role == "assistant":
        session["last_answer_summary"] = clean_text(summary)

    return session


def append_user_turn(session: Dict[str, Any], question: str) -> Dict[str, Any]:
    return append_turn(
        session=session,
        role="user",
        content=question,
        summary=truncate(clean_text(question), 120),
    )


def append_assistant_turn(
    session: Dict[str, Any],
    answer_md: str,
    summary: str = "",
    problem_ids: Optional[List[str]] = None,
    problem_names: Optional[List[str]] = None,
    retrieval_result_path: str = "",
    answer_path: str = "",
    answer_result_path: str = "",
) -> Dict[str, Any]:
    if not summary:
        summary = summarize_answer_markdown(answer_md)

    return append_turn(
        session=session,
        role="assistant",
        content=answer_md,
        summary=summary,
        problem_ids=problem_ids,
        problem_names=problem_names,
        retrieval_result_path=retrieval_result_path,
        answer_path=answer_path,
        answer_result_path=answer_result_path,
    )


def truncate(text: str, max_chars: int = 200) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def extract_section(markdown: str, section_title: str) -> str:
    markdown = markdown or ""
    pattern = re.escape(section_title)
    match = re.search(pattern, markdown)
    if not match:
        return ""

    start = match.end()
    rest = markdown[start:]

    next_match = re.search(r"\n[一二三四五六七八九十]+、", rest)
    if next_match:
        return rest[:next_match.start()].strip()

    return rest.strip()


def summarize_answer_markdown(answer_md: str) -> str:
    """
    Rule-based summary for memory.
    Keep it short because the session memory will be reused in later prompts.
    """
    answer_md = clean_text(answer_md)

    if not answer_md:
        return ""

    understanding = extract_section(answer_md, "一、我理解的照护问题")
    action = extract_section(answer_md, "二、可以先这样做")
    risk = extract_section(answer_md, "五、风险提醒")
    consult = extract_section(answer_md, "六、什么时候需要咨询专业人员")

    pieces: List[str] = []

    if understanding:
        pieces.append("照护问题：" + truncate(understanding, 180))
    if action:
        pieces.append("主要建议：" + truncate(action, 220))
    if risk:
        pieces.append("风险提醒：" + truncate(risk, 160))
    if consult:
        pieces.append("就医/咨询：" + truncate(consult, 160))

    if not pieces:
        return truncate(answer_md, 360)

    return "；".join(pieces)


def update_active_problems(
    session: Dict[str, Any],
    selected_problems: Optional[List[Dict[str, Any]]] = None,
    problem_ids: Optional[List[str]] = None,
    problem_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    selected_problems = selected_problems or []

    active_ids = normalize_problem_ids(session.get("active_problem_ids") or [])
    active_map: Dict[str, Dict[str, Any]] = {}

    for p in session.get("active_problems") or []:
        pid = clean_text(p.get("id")).upper()
        if pid:
            active_map[pid] = p

    # From selected_problems.
    for p in selected_problems:
        pid = clean_text(p.get("id")).upper()
        if not pid:
            continue

        if pid not in active_ids:
            active_ids.append(pid)

        existing = active_map.get(pid, {})

        count = int(existing.get("count", 0) or 0) + 1

        active_map[pid] = {
            "id": pid,
            "name": clean_text(p.get("name") or p.get("content") or existing.get("name")),
            "content": clean_text(p.get("content") or p.get("name") or existing.get("content")),
            "category": clean_text(p.get("category") or existing.get("category")),
            "first_seen": existing.get("first_seen") or now_iso(),
            "last_seen": now_iso(),
            "count": count,
            "confidence": p.get("confidence", existing.get("confidence")),
            "last_evidence_in_question": clean_text(p.get("evidence_in_question") or existing.get("last_evidence_in_question")),
            "last_reason": clean_text(p.get("reason") or existing.get("last_reason")),
        }

    # From plain IDs/names.
    ids = normalize_problem_ids(problem_ids)
    names = problem_names or []

    for idx, pid in enumerate(ids):
        if pid not in active_ids:
            active_ids.append(pid)

        existing = active_map.get(pid, {})
        name = clean_text(names[idx] if idx < len(names) else "")

        count = int(existing.get("count", 0) or 0) + 1

        active_map[pid] = {
            "id": pid,
            "name": clean_text(existing.get("name") or name),
            "content": clean_text(existing.get("content") or name),
            "category": clean_text(existing.get("category")),
            "first_seen": existing.get("first_seen") or now_iso(),
            "last_seen": now_iso(),
            "count": count,
            "confidence": existing.get("confidence"),
            "last_evidence_in_question": existing.get("last_evidence_in_question", ""),
            "last_reason": existing.get("last_reason", ""),
        }

    session["active_problem_ids"] = active_ids
    session["active_problems"] = [active_map[pid] for pid in active_ids if pid in active_map]

    update_risk_flags(session)

    return session


def update_risk_flags(session: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = list(session.get("risk_flags") or [])

    def add_flag(flag: str) -> None:
        flag = clean_text(flag)
        if flag and flag not in flags:
            flags.append(flag)

    for p in session.get("active_problems") or []:
        pid = clean_text(p.get("id")).upper()
        name = clean_text(p.get("name") or p.get("content"))

        if pid in RISK_PROBLEM_IDS:
            add_flag(f"{pid}:{name}")

        for kw in RISK_KEYWORDS:
            if kw in name:
                add_flag(f"{pid}:{name}")
                break

    # Also scan recent user descriptions.
    profile = session.get("care_recipient_profile") or {}
    for desc in profile.get("recent_user_descriptions") or []:
        for kw in RISK_KEYWORDS:
            if kw in desc:
                add_flag(f"文本风险线索:{kw}")

    session["risk_flags"] = flags[-20:]

    return session


def update_from_retrieval_result(
    session: Dict[str, Any],
    retrieval_result: Dict[str, Any],
    retrieval_result_path: str = "",
) -> Dict[str, Any]:
    question = clean_text(retrieval_result.get("question"))

    selected_problems = retrieval_result.get("selected_problems") or []
    selected_problem_ids = normalize_problem_ids(retrieval_result.get("selected_problem_ids") or [])

    problem_names = [
        clean_text(p.get("content") or p.get("name"))
        for p in selected_problems
        if clean_text(p.get("content") or p.get("name"))
    ]

    if question:
        append_user_turn(session, question)

    update_active_problems(
        session=session,
        selected_problems=selected_problems,
        problem_ids=selected_problem_ids,
        problem_names=problem_names,
    )

    if retrieval_result_path:
        session["last_retrieval_result_path"] = retrieval_result_path

    return session


def update_from_answer_result(
    session: Dict[str, Any],
    answer_result: Dict[str, Any],
    answer_result_path: str = "",
    answer_md_path: str = "",
    retrieval_result_path: str = "",
) -> Dict[str, Any]:
    question = clean_text(answer_result.get("question"))
    answer_md = clean_text(answer_result.get("answer_md"))

    selected_problems = answer_result.get("selected_problems") or []
    selected_problem_ids = normalize_problem_ids(answer_result.get("selected_problem_ids") or [])

    problem_names = [
        clean_text(p.get("content") or p.get("name"))
        for p in selected_problems
        if clean_text(p.get("content") or p.get("name"))
    ]

    if question:
        append_user_turn(session, question)

    update_active_problems(
        session=session,
        selected_problems=selected_problems,
        problem_ids=selected_problem_ids,
        problem_names=problem_names,
    )

    if answer_md:
        append_assistant_turn(
            session=session,
            answer_md=answer_md,
            problem_ids=selected_problem_ids,
            problem_names=problem_names,
            retrieval_result_path=retrieval_result_path,
            answer_path=answer_md_path,
            answer_result_path=answer_result_path,
        )

    if retrieval_result_path:
        session["last_retrieval_result_path"] = retrieval_result_path
    if answer_md_path:
        session["last_answer_path"] = answer_md_path
    if answer_result_path:
        session["last_answer_result_path"] = answer_result_path

    return session


def generate_conversation_summary(session: Dict[str, Any]) -> str:
    profile = session.get("care_recipient_profile") or {}
    active_problems = session.get("active_problems") or []
    risk_flags = session.get("risk_flags") or []
    turns = session.get("conversation_turns") or []

    display_name = clean_text(profile.get("display_name") or profile.get("relation") or "老人")
    relation = clean_text(profile.get("relation"))
    living_context = clean_text(profile.get("living_context"))
    notes = profile.get("notes") or []
    recent_desc = profile.get("recent_user_descriptions") or []

    lines: List[str] = []

    lines.append("【会话摘要】")
    lines.append(f"- 照护对象：{display_name}" + (f"（关系：{relation}）" if relation and relation != display_name else ""))

    if living_context:
        lines.append(f"- 居住/照护背景：{living_context}")

    if notes:
        lines.append("- 其他备注：" + "；".join(clean_text(x) for x in notes[-5:] if clean_text(x)))

    if recent_desc:
        lines.append("- 家属最近描述：" + "；".join(clean_text(x) for x in recent_desc[-4:] if clean_text(x)))

    if active_problems:
        lines.append("- 已识别照护问题：")
        for p in active_problems:
            pid = clean_text(p.get("id"))
            category = clean_text(p.get("category"))
            name = clean_text(p.get("content") or p.get("name"))
            count = p.get("count", 1)
            lines.append(f"  - {pid} [{category}] {name}（出现/提及 {count} 次）")
    else:
        lines.append("- 已识别照护问题：暂无")

    if risk_flags:
        lines.append("- 需要持续关注的风险线索：" + "；".join(risk_flags[-8:]))
    else:
        lines.append("- 需要持续关注的风险线索：暂无明确记录")

    recent_turns = turns[-MAX_RECENT_TURNS_IN_SUMMARY:]
    if recent_turns:
        lines.append("- 最近对话：")
        for turn in recent_turns:
            role = turn.get("role")
            if role == "user":
                lines.append(f"  - 家属问：{truncate(turn.get('content', ''), 120)}")
            elif role == "assistant":
                summary = clean_text(turn.get("summary")) or summarize_answer_markdown(turn.get("content", ""))
                lines.append(f"  - 系统答复摘要：{truncate(summary, 180)}")
            else:
                lines.append(f"  - {role}: {truncate(turn.get('content', ''), 120)}")

    lines.append("【使用原则】该摘要只用于理解多轮上下文；每一轮正式回答仍应重新进行问题映射、图谱检索和证据约束生成。")

    summary = "\n".join(lines)

    session["conversation_summary"] = summary
    session["updated_at"] = now_iso()

    return summary


def update_from_last_outputs(session: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use latest Step 06 answer result if available.
    Fallback to latest Step 05 retrieval result if answer result is not available.
    """
    answer_path = LAST_ANSWER_JSON_PATH
    answer_md_path = LAST_ANSWER_MD_PATH
    retrieval_path = LAST_RETRIEVAL_RESULT_PATH

    if answer_path.exists():
        answer_result = load_json_file(answer_path)

        update_from_answer_result(
            session=session,
            answer_result=answer_result,
            answer_result_path=relative_path(answer_path),
            answer_md_path=relative_path(answer_md_path),
            retrieval_result_path=relative_path(retrieval_path) if retrieval_path.exists() else "",
        )

    elif retrieval_path.exists():
        retrieval_result = load_json_file(retrieval_path)

        update_from_retrieval_result(
            session=session,
            retrieval_result=retrieval_result,
            retrieval_result_path=relative_path(retrieval_path),
        )

    else:
        raise FileNotFoundError(
            "No last output found. Please run Step 05 or Step 06 first."
        )

    generate_conversation_summary(session)

    return session


def list_sessions() -> List[Dict[str, Any]]:
    ensure_dirs()

    items: List[Dict[str, Any]] = []

    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            data = load_json_file(path)
            items.append(
                {
                    "session_id": data.get("session_id"),
                    "updated_at": data.get("updated_at"),
                    "active_problem_ids": data.get("active_problem_ids"),
                    "turn_count": len(data.get("conversation_turns") or []),
                    "path": relative_path(path),
                }
            )
        except Exception:
            continue

    return items


def print_session_preview(session: Dict[str, Any]) -> None:
    print_header("Session preview")
    print(f"Session ID: {session.get('session_id')}")
    print(f"Created at: {session.get('created_at')}")
    print(f"Updated at: {session.get('updated_at')}")

    print("\nCare recipient profile:")
    print(json.dumps(session.get("care_recipient_profile"), ensure_ascii=False, indent=2))

    print("\nActive problem IDs:")
    print(session.get("active_problem_ids"))

    print("\nActive problems:")
    for p in session.get("active_problems") or []:
        print(f"- {p.get('id')} [{p.get('category')}]: {p.get('content') or p.get('name')} count={p.get('count')}")

    print("\nRisk flags:")
    for flag in session.get("risk_flags") or []:
        print(f"- {flag}")

    print("\nRecent turns:")
    for turn in (session.get("conversation_turns") or [])[-6:]:
        role = turn.get("role")
        if role == "assistant":
            print(f"- assistant: {truncate(turn.get('summary') or turn.get('content'), 180)}")
        else:
            print(f"- {role}: {truncate(turn.get('content'), 180)}")

    print("\nConversation summary:")
    print(session.get("conversation_summary") or "(empty)")


def parse_profile_json(profile_json: str) -> Dict[str, Any]:
    profile_json = clean_text(profile_json)
    if not profile_json:
        return {}

    try:
        data = json.loads(profile_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            "Invalid --profile_json. Example:\n"
            """--profile_json '{"display_name":"奶奶","relation":"奶奶","living_context":"与家属同住"}'"""
        ) from e

    if not isinstance(data, dict):
        raise ValueError("--profile_json must be a JSON object.")

    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 07: Local session memory for multi-turn GraphRAG agent."
    )

    parser.add_argument("--session_id", type=str, default=DEFAULT_SESSION_ID)

    parser.add_argument("--create", action="store_true", help="Create session if missing.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing session with backup.")
    parser.add_argument("--show", action="store_true", help="Show session preview.")
    parser.add_argument("--summary", action="store_true", help="Generate and print conversation summary.")
    parser.add_argument("--list", action="store_true", help="List existing sessions.")

    parser.add_argument(
        "--append_user",
        type=str,
        default="",
        help="Append a user turn.",
    )

    parser.add_argument(
        "--append_assistant",
        type=str,
        default="",
        help="Append an assistant turn.",
    )

    parser.add_argument(
        "--assistant_summary",
        type=str,
        default="",
        help="Optional summary for --append_assistant.",
    )

    parser.add_argument(
        "--problem_ids",
        nargs="*",
        default=None,
        help="Optional problem IDs to update memory, e.g. --problem_ids F2 H14",
    )

    parser.add_argument(
        "--problem_names",
        nargs="*",
        default=None,
        help="Optional problem names corresponding to --problem_ids.",
    )

    parser.add_argument(
        "--profile_json",
        type=str,
        default="",
        help='Update care recipient profile. Example: --profile_json \'{"display_name":"奶奶","relation":"奶奶"}\'',
    )

    parser.add_argument(
        "--update_from_last",
        action="store_true",
        help="Update memory from outputs/answers/last_answer_result.json or outputs/retrieval_debug/last_retrieval_result.json.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset this session. Existing session will be backed up.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header("Step 07: Session memory")
    print("[LOCAL] This script only reads/writes local session JSON files.")

    if args.list:
        items = list_sessions()
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return

    if args.reset:
        session = create_session(args.session_id, overwrite=True)
        print(f"[OK] Session reset: {session_path(args.session_id)}")
        print_session_preview(session)
        return

    if args.create or args.overwrite:
        session = create_session(args.session_id, overwrite=args.overwrite)
    else:
        session = load_session(args.session_id, create_if_missing=True)

    if args.profile_json:
        profile_updates = parse_profile_json(args.profile_json)
        update_care_recipient_profile(session, profile_updates=profile_updates)

    if args.append_user:
        append_user_turn(session, args.append_user)

    if args.append_assistant:
        append_assistant_turn(
            session=session,
            answer_md=args.append_assistant,
            summary=args.assistant_summary,
            problem_ids=args.problem_ids,
            problem_names=args.problem_names,
        )

    if args.problem_ids:
        update_active_problems(
            session=session,
            problem_ids=args.problem_ids,
            problem_names=args.problem_names,
        )

    if args.update_from_last:
        update_from_last_outputs(session)

    if args.summary:
        summary = generate_conversation_summary(session)
        print_header("Conversation summary")
        print(summary)

    # Always refresh summary after meaningful updates.
    if (
        args.append_user
        or args.append_assistant
        or args.problem_ids
        or args.profile_json
        or args.update_from_last
    ):
        generate_conversation_summary(session)

    path = save_session(session)

    print_header("Final result")
    print("[PASS] Step 07 session memory updated.")
    print(f"[OK] Session path: {path}")

    if args.show or args.summary or args.update_from_last or args.create or args.overwrite:
        print_session_preview(session)


if __name__ == "__main__":
    main()
