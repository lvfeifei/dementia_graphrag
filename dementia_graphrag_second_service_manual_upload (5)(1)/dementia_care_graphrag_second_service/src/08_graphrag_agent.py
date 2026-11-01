"""
Step 08: Unified multi-turn GraphRAG agent.

This script connects:

03_llm_client.py
04_problem_mapper.py
05_retrieve.py
06_generate_answer.py
07_session_memory.py

Main workflow:
1. Load or create session memory
2. Read current user question
3. Build memory-aware query when needed
4. Run retrieval:
   - problem mapping
   - Neo4j graph retrieval
   - FAISS Advice retrieval
   - keyword retrieval
   - fusion
5. Generate evidence-grounded answer
6. Update session memory
7. Save answer and retrieval outputs

This script does NOT modify Neo4j.
It only reads KG/indexes and writes local output/session files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

DEFAULT_SESSION_ID = "demo_001"

_RUNTIME_MODULES_CACHE = None

PRONOUN_OR_CONTEXT_WORDS = [
    "她", "他", "它", "这个", "那个", "这", "那",
    "刚才", "上次", "刚刚", "前面", "之前", "继续",
    "钱", "没丢", "解释", "有用吗", "怎么办", "还",
    "晚上", "白天", "门", "出去", "睡觉", "打人",
]


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


def load_module(module_filename: str, module_name: str):
    module_path = SRC_DIR / module_filename

    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_runtime_modules() -> Dict[str, Any]:
    """
    Cache Step 05/06/07 modules in the current Python process.

    This is important for Streamlit:
    Streamlit itself reruns the script often, but if streamlit_app.py caches
    the Step 08 module, then this runtime cache keeps Step 05 alive too.
    Step 05 can then reuse BGE-M3, FAISS and mapper caches.
    """
    global _RUNTIME_MODULES_CACHE

    if _RUNTIME_MODULES_CACHE is not None:
        return _RUNTIME_MODULES_CACHE

    retrieve_module = load_module("05_retrieve.py", "step05_retrieve")
    answer_module = load_module("06_generate_answer.py", "step06_generate_answer")
    memory_module = load_module("07_session_memory.py", "step07_session_memory")

    _RUNTIME_MODULES_CACHE = {
        "retrieve": retrieve_module,
        "answer": answer_module,
        "memory": memory_module,
    }

    return _RUNTIME_MODULES_CACHE


def run_optional_tools(
    make_pdf: bool,
    make_ics: bool,
    ics_days: int,
    answer_result_path: str,
    retrieval_result_path: str,
) -> Dict[str, Any]:
    """
    Optional tool calling layer.

    PDF reads answer_result JSON.
    ICS reads retrieval_result JSON.
    """
    tool_outputs: Dict[str, Any] = {
        "pdf": None,
        "ics": None,
    }

    if make_pdf:
        pdf_module = load_module("09_pdf_generator.py", "step09_pdf_generator")
        pdf_result = pdf_module.generate_pdf_from_answer_result(Path(answer_result_path))
        tool_outputs["pdf"] = pdf_result

    if make_ics:
        ics_module = load_module("10_calendar_generator.py", "step10_calendar_generator")
        ics_result = ics_module.generate_calendar_from_retrieval_result(
            retrieval_result_path=Path(retrieval_result_path),
            days=ics_days,
        )
        tool_outputs["ics"] = ics_result

    return tool_outputs


def relative_path(path_like: Any) -> str:
    if not path_like:
        return ""

    path = Path(path_like)

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def get_profile_text(session: Dict[str, Any]) -> str:
    profile = session.get("care_recipient_profile") or {}

    display_name = clean_text(profile.get("display_name") or profile.get("relation") or "老人")
    relation = clean_text(profile.get("relation"))
    living_context = clean_text(profile.get("living_context"))
    main_caregiver = clean_text(profile.get("main_caregiver"))
    notes = profile.get("notes") or []
    recent_desc = profile.get("recent_user_descriptions") or []

    lines: List[str] = []
    lines.append(f"照护对象：{display_name}")

    if relation and relation != display_name:
        lines.append(f"与照护者关系：{relation}")
    if living_context:
        lines.append(f"居住/照护背景：{living_context}")
    if main_caregiver:
        lines.append(f"主要照护者：{main_caregiver}")
    if notes:
        lines.append("备注：" + "；".join(clean_text(x) for x in notes[-5:] if clean_text(x)))
    if recent_desc:
        lines.append("近期家属描述：" + "；".join(clean_text(x) for x in recent_desc[-4:] if clean_text(x)))

    return "\n".join(lines)


def get_active_problem_text(session: Dict[str, Any]) -> str:
    active_problems = session.get("active_problems") or []

    if not active_problems:
        return "暂无已记录的照护问题。"

    lines: List[str] = []

    for p in active_problems:
        pid = clean_text(p.get("id"))
        category = clean_text(p.get("category"))
        name = clean_text(p.get("content") or p.get("name"))
        count = p.get("count", 1)
        lines.append(f"- {pid} [{category}] {name}（既往提及 {count} 次）")

    return "\n".join(lines)


def is_memory_context_needed(question: str, session: Dict[str, Any], mode: str = "auto") -> bool:
    """
    mode:
    - auto: use memory only when current question looks context-dependent
    - always: always use session memory
    - off: never use memory
    """
    question = clean_text(question)

    if mode == "off":
        return False

    if mode == "always":
        return bool(session.get("conversation_summary") or session.get("active_problem_ids"))

    # auto
    if not session.get("conversation_summary") and not session.get("active_problem_ids"):
        return False

    if len(question) <= 35:
        return True

    if any(word in question for word in PRONOUN_OR_CONTEXT_WORDS):
        return True

    # If the user asks a short follow-up-like question.
    if question.endswith(("吗？", "吗", "怎么办？", "怎么办", "可以吗？", "可以吗")) and len(question) <= 70:
        return True

    return False


def build_memory_aware_retrieval_query(
    question: str,
    session: Dict[str, Any],
    memory_context_mode: str = "auto",
) -> Tuple[str, bool]:
    """
    Return:
    - retrieval_query: used for problem mapping and FAISS Advice retrieval
    - used_memory: whether memory context was included

    Important:
    This context is only used to resolve pronouns / short follow-up questions.
    The final answer still uses evidence-grounded retrieval results.
    """
    question = clean_text(question)
    use_memory = is_memory_context_needed(question, session, mode=memory_context_mode)

    if not use_memory:
        return question, False

    conversation_summary = clean_text(session.get("conversation_summary"))
    profile_text = get_profile_text(session)
    active_problem_text = get_active_problem_text(session)
    risk_flags = session.get("risk_flags") or []

    risk_text = "；".join(clean_text(x) for x in risk_flags[-8:] if clean_text(x)) or "暂无明确风险线索。"

    retrieval_query = f"""
当前照护者问题：
{question}

以下是本会话已有背景，仅用于理解指代和上下文，不代表本轮一定要全部使用：
{profile_text}

既往已识别照护问题：
{active_problem_text}

既往风险线索：
{risk_text}

会话摘要：
{conversation_summary}

请结合以上背景理解当前问题中的“她/他/这个/那/钱/晚上/刚才”等指代，但本轮仍应根据当前问题重新判断最相关的照护问题。
""".strip()

    return retrieval_query, True


def build_answer_display_question(
    question: str,
    session: Dict[str, Any],
    used_memory: bool,
) -> str:
    """
    Build a concise question text shown to the answer generator.
    This helps the model resolve pronouns without dumping the entire memory summary.
    """
    question = clean_text(question)

    if not used_memory:
        return question

    profile = session.get("care_recipient_profile") or {}
    display_name = clean_text(profile.get("display_name") or profile.get("relation") or "老人")
    active_problem_text = get_active_problem_text(session)

    return f"""
照护者当前追问：
{question}

已知会话背景：
照护对象是{display_name}。此前会话中已记录的相关照护问题包括：
{active_problem_text}

请优先回答当前追问，不要机械重复上一轮全部内容。
""".strip()


def maybe_merge_active_problem_ids(
    selected_ids: List[str],
    session: Dict[str, Any],
    include_active_problem_ids: bool,
) -> List[str]:
    """
    Usually we should not blindly carry all active problems into every round,
    because it can pollute retrieval.

    But when user explicitly wants continuity, or asks a very short follow-up,
    caller can enable this.
    """
    selected_ids = normalize_problem_ids(selected_ids)

    if not include_active_problem_ids:
        return selected_ids

    active_ids = normalize_problem_ids(session.get("active_problem_ids") or [])
    merged = list(selected_ids)

    for pid in active_ids:
        if pid not in merged:
            merged.append(pid)

    return merged


def extract_problem_names(selected_problems: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []

    for p in selected_problems or []:
        name = clean_text(p.get("content") or p.get("name"))
        if name:
            names.append(name)

    return names


def run_agent(
    session_id: str,
    question: str,
    problem_ids: Optional[List[str]] = None,
    memory_context_mode: str = "auto",
    include_active_problem_ids: bool = False,
    reset_session: bool = False,
    no_auto_map: bool = False,
    no_debug: bool = False,
    temperature: float = 0.2,
    max_tokens: int = 3200,
    show_steps: bool = True,
    make_pdf: bool = False,
    make_ics: bool = False,
    ics_days: int = 7,
) -> Dict[str, Any]:
    modules = load_runtime_modules()
    retrieve_module = modules["retrieve"]
    answer_module = modules["answer"]
    memory_module = modules["memory"]

    question = clean_text(question)
    if not question:
        raise ValueError("question is empty.")

    if reset_session:
        session = memory_module.create_session(session_id, overwrite=True)
    else:
        session = memory_module.load_session(session_id, create_if_missing=True)

    # Record the user's original wording first.
    memory_module.append_user_turn(session, question)
    memory_module.update_care_recipient_profile(session, source_text=question)
    memory_module.generate_conversation_summary(session)
    memory_module.save_session(session)

    retrieval_query, used_memory_context = build_memory_aware_retrieval_query(
        question=question,
        session=session,
        memory_context_mode=memory_context_mode,
    )

    if show_steps:
        print_header("Agent context")
        print(f"Session ID: {session_id}")
        print(f"Used memory context: {used_memory_context}")
        print(f"Memory mode: {memory_context_mode}")
        print(f"Manual problem IDs: {problem_ids or []}")

    manual_problem_ids = normalize_problem_ids(problem_ids)

    # If user gave problem IDs, retrieve directly.
    # If not, Step 05 will auto-call 04_problem_mapper.py unless no_auto_map=True.
    retrieval_result = retrieve_module.retrieve(
        question=retrieval_query,
        problem_ids=manual_problem_ids,
        auto_map=not no_auto_map,
        include_debug=not no_debug,
    )

    # Optionally merge active problem IDs only after current mapping.
    # This is useful for short follow-ups, but disabled by default to avoid pollution.
    if include_active_problem_ids:
        current_ids = normalize_problem_ids(retrieval_result.get("selected_problem_ids") or [])
        merged_ids = maybe_merge_active_problem_ids(
            selected_ids=current_ids,
            session=session,
            include_active_problem_ids=True,
        )

        if merged_ids != current_ids:
            if show_steps:
                print_header("Active problem merge")
                print(f"Current IDs: {current_ids}")
                print(f"Merged IDs: {merged_ids}")

            retrieval_result = retrieve_module.retrieve(
                question=retrieval_query,
                problem_ids=merged_ids,
                auto_map=False,
                include_debug=not no_debug,
            )

    # Keep internal trace.
    retrieval_result["original_question"] = question
    retrieval_result["memory_context_used"] = used_memory_context
    retrieval_result["memory_context_mode"] = memory_context_mode
    retrieval_result["session_id"] = session_id

    # Replace question for answer generation with a concise display version.
    # The long retrieval query remains available in retrieval_query_for_trace.
    retrieval_result["retrieval_query_for_trace"] = retrieval_query
    retrieval_result["question"] = build_answer_display_question(
        question=question,
        session=session,
        used_memory=used_memory_context,
    )

    retrieval_last_path, retrieval_timestamp_path = retrieve_module.write_retrieval_result(retrieval_result)

    if show_steps:
        print_header("Retrieval result")
        print(f"Selected problem IDs: {retrieval_result.get('selected_problem_ids')}")
        print("Retrieval counts:")
        print(json.dumps(retrieval_result.get("retrieval_counts"), ensure_ascii=False, indent=2))
        print(f"Retrieval saved: {retrieval_timestamp_path}")

    answer_result = answer_module.generate_answer(
        retrieval_result=retrieval_result,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer_paths = answer_module.write_answer_result(answer_result)

    tool_outputs = run_optional_tools(
        make_pdf=make_pdf,
        make_ics=make_ics,
        ics_days=ics_days,
        answer_result_path=answer_paths.get("timestamp_answer_json"),
        retrieval_result_path=str(retrieval_timestamp_path),
    )

    if show_steps:
        print_header("Answer")
        print(answer_result.get("answer_md", ""))

        if make_pdf or make_ics:
            print_header("Tool outputs")
            print(json.dumps(tool_outputs, ensure_ascii=False, indent=2))

    # Update session memory from original user question + answer.
    selected_problems = answer_result.get("selected_problems") or retrieval_result.get("selected_problems") or []
    selected_problem_ids = normalize_problem_ids(
        answer_result.get("selected_problem_ids") or retrieval_result.get("selected_problem_ids") or []
    )
    problem_names = extract_problem_names(selected_problems)

    memory_module.update_active_problems(
        session=session,
        selected_problems=selected_problems,
        problem_ids=selected_problem_ids,
        problem_names=problem_names,
    )

    memory_module.append_assistant_turn(
        session=session,
        answer_md=answer_result.get("answer_md", ""),
        problem_ids=selected_problem_ids,
        problem_names=problem_names,
        retrieval_result_path=relative_path(retrieval_timestamp_path),
        answer_path=relative_path(answer_paths.get("timestamp_answer_md")),
        answer_result_path=relative_path(answer_paths.get("timestamp_answer_json")),
    )

    session["last_user_question"] = question
    session["last_retrieval_result_path"] = relative_path(retrieval_timestamp_path)
    session["last_answer_path"] = relative_path(answer_paths.get("timestamp_answer_md"))
    session["last_answer_result_path"] = relative_path(answer_paths.get("timestamp_answer_json"))

    memory_module.generate_conversation_summary(session)
    session_path = memory_module.save_session(session)

    final_result = {
        "success": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v02_final_workflow_step08_agent",
        "session_id": session_id,
        "question": question,
        "used_memory_context": used_memory_context,
        "memory_context_mode": memory_context_mode,
        "selected_problem_ids": selected_problem_ids,
        "retrieval_counts": retrieval_result.get("retrieval_counts"),
        "answer_md": answer_result.get("answer_md"),
        "paths": {
            "session": str(session_path),
            "retrieval_last": str(retrieval_last_path),
            "retrieval_timestamp": str(retrieval_timestamp_path),
            "answer_last_md": answer_paths.get("last_answer_md"),
            "answer_last_json": answer_paths.get("last_answer_json"),
            "answer_timestamp_md": answer_paths.get("timestamp_answer_md"),
            "answer_timestamp_json": answer_paths.get("timestamp_answer_json"),
        },
        "tool_outputs": tool_outputs,
        "session_summary": session.get("conversation_summary"),
    }

    return final_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 08: Unified multi-turn GraphRAG agent."
    )

    parser.add_argument(
        "--session_id",
        type=str,
        default=DEFAULT_SESSION_ID,
        help="Session ID, e.g. demo_001.",
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question.",
    )

    parser.add_argument(
        "--problem_ids",
        nargs="*",
        default=None,
        help="Optional manual Problem IDs, e.g. --problem_ids F2 H14.",
    )

    parser.add_argument(
        "--memory_context_mode",
        choices=["auto", "always", "off"],
        default="auto",
        help="Whether to include session memory in query understanding.",
    )

    parser.add_argument(
        "--include_active_problem_ids",
        action="store_true",
        help="Merge existing active_problem_ids into retrieval. Use carefully for short follow-ups.",
    )

    parser.add_argument(
        "--reset_session",
        action="store_true",
        help="Reset session before answering. Existing session will be backed up.",
    )

    parser.add_argument(
        "--no_auto_map",
        action="store_true",
        help="Disable auto problem mapping when problem_ids are not provided.",
    )

    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="Do not include retrieval debug details.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=3200,
    )

    parser.add_argument(
        "--make_pdf",
        action="store_true",
        help="Generate PDF care plan after answer generation.",
    )

    parser.add_argument(
        "--make_ics",
        action="store_true",
        help="Generate ICS calendar reminders after answer generation.",
    )

    parser.add_argument(
        "--ics_days",
        type=int,
        default=7,
        help="Number of days for recurring ICS reminders.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print full answer preview.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header("Step 08: Unified multi-turn GraphRAG agent")
    print("[READ-ONLY] This script reads KG/indexes and writes local session/output files. It does not modify KG.")
    print(f"Question: {args.question}")

    result = run_agent(
        session_id=args.session_id,
        question=args.question,
        problem_ids=args.problem_ids,
        memory_context_mode=args.memory_context_mode,
        include_active_problem_ids=args.include_active_problem_ids,
        reset_session=args.reset_session,
        no_auto_map=args.no_auto_map,
        no_debug=args.no_debug,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        show_steps=not args.quiet,
        make_pdf=args.make_pdf,
        make_ics=args.make_ics,
        ics_days=args.ics_days,
    )

    print_header("Final result")
    print("[PASS] Step 08 agent run finished.")
    print(f"[OK] Session: {result['paths']['session']}")
    print(f"[OK] Retrieval: {result['paths']['retrieval_timestamp']}")
    print(f"[OK] Answer markdown: {result['paths']['answer_timestamp_md']}")
    print(f"[OK] Answer JSON: {result['paths']['answer_timestamp_json']}")
    print(f"[OK] Selected Problem IDs: {result.get('selected_problem_ids')}")
    print(f"[OK] Used memory context: {result.get('used_memory_context')}")
    if result.get("tool_outputs"):
        print(f"[OK] Tool outputs: {json.dumps(result.get('tool_outputs'), ensure_ascii=False)}")
    print("Next step: build app.py or Streamlit demo.")


if __name__ == "__main__":
    main()
