"""
Step 06: Generate evidence-grounded care answer.

This module reads the retrieval context from Step 05 and asks DeepSeek to generate
a caregiver-facing answer in a fixed 8-section structure:

一、我理解的照护问题
二、可以先这样做
三、白天和夜间活动安排
四、环境调整建议
五、风险提醒
六、什么时候需要咨询专业人员
七、依据来源
八、说明：本回答用于照护教育支持，不能替代专业诊疗

Inputs:
1. outputs/retrieval_debug/last_retrieval_result.json by default
2. Or a custom retrieval result JSON path
3. Or a new question, then this script calls src/05_retrieve.py first

Outputs:
1. outputs/answers/last_answer.md
2. outputs/answers/last_answer_result.json
3. outputs/answers/answer_<timestamp>.md
4. outputs/answers/answer_result_<timestamp>.json

This script does NOT modify Neo4j.
It only reads retrieval results and calls DeepSeek.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

RETRIEVAL_DEBUG_DIR = PROJECT_ROOT / "outputs" / "retrieval_debug"
ANSWERS_DIR = PROJECT_ROOT / "outputs" / "answers"

DEFAULT_RETRIEVAL_RESULT_PATH = RETRIEVAL_DEBUG_DIR / "last_retrieval_result.json"

LAST_ANSWER_MD_PATH = ANSWERS_DIR / "last_answer.md"
LAST_ANSWER_JSON_PATH = ANSWERS_DIR / "last_answer_result.json"

DEFAULT_MAX_PROBLEM_ADVICE = 8
DEFAULT_MAX_ACTIVITY_ADVICE = 5
DEFAULT_MAX_ENVIRONMENT_ADVICE = 5
DEFAULT_MAX_EVIDENCE_SOURCES = 10


REQUIRED_SECTIONS = [
    "一、我理解的照护问题",
    "二、可以先这样做",
    "三、白天和夜间活动安排",
    "四、环境调整建议",
    "五、风险提醒",
    "六、什么时候需要咨询专业人员",
    "七、依据来源",
    "八、说明：本回答用于照护教育支持，不能替代专业诊疗",
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


def ensure_dirs() -> None:
    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Retrieval result not found: {path}\n"
            "Please run Step 05 first, for example:\n"
            "python src/05_retrieve.py --question \"老人白天睡很多，晚上特别精神，还一直走来走去。\" --problem_ids F3 H5"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"Retrieval result should be a JSON object: {path}")

    return data


def load_deepseek_client_class():
    module_path = SRC_DIR / "03_llm_client.py"

    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find {module_path}. Please create src/03_llm_client.py first."
        )

    spec = importlib.util.spec_from_file_location("step03_llm_client", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.DeepSeekClient


def load_retrieve_function():
    module_path = SRC_DIR / "05_retrieve.py"

    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find {module_path}. Please create src/05_retrieve.py first."
        )

    spec = importlib.util.spec_from_file_location("step05_retrieve", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.retrieve, module.write_retrieval_result


def normalize_problem_ids(problem_ids: Optional[List[str]]) -> List[str]:
    if not problem_ids:
        return []

    result: List[str] = []
    seen = set()

    for pid in problem_ids:
        pid = clean_text(pid).upper()
        if not pid:
            continue
        if pid not in seen:
            result.append(pid)
            seen.add(pid)

    return result


def truncate_text(text: Any, max_chars: int = 360) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def item_line(item: Dict[str, Any], index: int, prefix: str) -> str:
    """
    Build compact advice line with traceable internal IDs.
    """
    tag = f"{prefix}{index}"
    doc_id = clean_text(item.get("doc_id"))
    node_label = clean_text(item.get("node_label"))
    advice_text = truncate_text(item.get("advice_text"), 420)

    related_problem_names = item.get("related_problem_names") or []
    related_problem_ids = item.get("related_problem_ids") or []
    problem_text = ""

    if related_problem_ids or related_problem_names:
        problem_text = f" | 关联问题: {related_problem_ids} {related_problem_names}"

    source_title = clean_text(item.get("source_title"))
    heading_path = clean_text(item.get("heading_path"))
    evidence_text = truncate_text(item.get("evidence_text"), 260)

    source_text = ""
    if source_title:
        source_text += f" | 来源: {source_title}"
    if heading_path:
        source_text += f" | 路径: {heading_path}"
    if evidence_text:
        source_text += f" | 证据摘录: {evidence_text}"

    return f"[{tag}] {doc_id} ({node_label}){problem_text} | 建议: {advice_text}{source_text}"


def source_line(source: Dict[str, Any], index: int) -> str:
    tag = f"S{index}"
    source_title = clean_text(source.get("source_title"))
    source_type = clean_text(source.get("source_type"))
    year = clean_text(source.get("year"))
    heading_path = clean_text(source.get("heading_path"))
    evidence_text = truncate_text(source.get("evidence_text"), 420)
    doc_id = clean_text(source.get("doc_id"))

    parts = [f"[{tag}]"]

    if source_title:
        parts.append(f"来源: {source_title}")
    if source_type:
        parts.append(f"类型: {source_type}")
    if year:
        parts.append(f"年份: {year}")
    if heading_path:
        parts.append(f"路径: {heading_path}")
    if doc_id:
        parts.append(f"对应知识条目: {doc_id}")
    if evidence_text:
        parts.append(f"证据摘录: {evidence_text}")

    return " | ".join(parts)


def build_context_for_prompt(
    retrieval_result: Dict[str, Any],
    max_problem_advice: int = DEFAULT_MAX_PROBLEM_ADVICE,
    max_activity_advice: int = DEFAULT_MAX_ACTIVITY_ADVICE,
    max_environment_advice: int = DEFAULT_MAX_ENVIRONMENT_ADVICE,
    max_evidence_sources: int = DEFAULT_MAX_EVIDENCE_SOURCES,
) -> Dict[str, Any]:
    """
    Convert retrieval_result.final_context into compact prompt context.
    """
    question = clean_text(retrieval_result.get("question"))
    selected_problems = retrieval_result.get("selected_problems") or []
    final_context = retrieval_result.get("final_context") or {}

    problem_advice = (final_context.get("problem_advice") or [])[:max_problem_advice]
    activity_advice = (final_context.get("activity_advice") or [])[:max_activity_advice]
    environment_advice = (final_context.get("environment_advice") or [])[:max_environment_advice]
    evidence_sources = (final_context.get("evidence_sources") or [])[:max_evidence_sources]

    selected_problem_lines: List[str] = []
    for p in selected_problems:
        pid = clean_text(p.get("id"))
        category = clean_text(p.get("category"))
        content = clean_text(p.get("content") or p.get("name"))
        confidence = p.get("confidence")
        evidence = clean_text(p.get("evidence_in_question"))

        line = f"- {pid} | 类别: {category} | 问题: {content}"
        if confidence is not None:
            line += f" | 映射置信度: {confidence}"
        if evidence:
            line += f" | 用户原文依据: {evidence}"
        selected_problem_lines.append(line)

    problem_advice_lines = [
        item_line(item, i + 1, "P")
        for i, item in enumerate(problem_advice)
    ]

    activity_advice_lines = [
        item_line(item, i + 1, "A")
        for i, item in enumerate(activity_advice)
    ]

    environment_advice_lines = [
        item_line(item, i + 1, "E")
        for i, item in enumerate(environment_advice)
    ]

    evidence_source_lines = [
        source_line(item, i + 1)
        for i, item in enumerate(evidence_sources)
    ]

    return {
        "question": question,
        "selected_problem_lines": selected_problem_lines,
        "problem_advice_lines": problem_advice_lines,
        "activity_advice_lines": activity_advice_lines,
        "environment_advice_lines": environment_advice_lines,
        "evidence_source_lines": evidence_source_lines,
        "retrieval_counts": retrieval_result.get("retrieval_counts") or {},
    }


def build_system_prompt() -> str:
    return """
你是一个面向失智老人居家照护者的中文照护教育支持助手。

你的任务：
基于系统提供的知识图谱检索结果，为照护者生成清晰、温和、可执行的照护建议。

必须遵守：
1. 只能依据“检索到的照护建议”和“证据来源”回答，不要编造未提供的专业知识、药物方案或诊断结论。
2. 你不是医生，不做诊断，不替代医生、护士或记忆门诊的专业判断。
3. 不要使用恐吓语气；要安抚照护者，但不能弱化安全风险。
4. 如果涉及跌倒、走失、攻击、自伤、误服、误吸、噎食、严重呛咳、持续幻觉妄想等风险，需要明确提醒何时寻求专业帮助。
5. 对照护者要说人话，避免学术腔；但要保留必要的专业边界。
6. 不要说“根据知识库”“根据上下文”这种机器话，可以说“依据检索到的指南/共识资料”。
7. 不要输出 JSON，只输出 Markdown。
8. 必须按照指定的八个一级标题输出，标题一个都不能少，也不要改标题。
9. 不要在正文中输出 [S1]、[P1]、[A1]、[E1] 这类引用编号。
10. 不要在正文中列出具体指南、共识或标准名称；这些内容会在界面的“查看本轮依据与证据”中单独展示。
11. “依据来源”部分只写一句简短提示：本轮建议基于知识图谱检索结果生成，具体依据可在回答下方展开查看。
12. 每条建议尽量具体，例如“先不要争辩”“陪她找一找物品”“白天增加光照”“夜间减少刺激”等。
13. 如果某一类建议检索结果不足，可以写“本次检索到的相关内容较少，可先重点执行前述建议”，不要硬编。
"""


def build_user_prompt(context: Dict[str, Any]) -> str:
    selected_problem_text = "\n".join(context["selected_problem_lines"]) or "无"
    problem_advice_text = "\n".join(context["problem_advice_lines"]) or "无"
    activity_advice_text = "\n".join(context["activity_advice_lines"]) or "无"
    environment_advice_text = "\n".join(context["environment_advice_lines"]) or "无"
    evidence_source_text = "\n".join(context["evidence_source_lines"]) or "无"

    return f"""
照护者原始问题：
{context["question"]}

系统识别到的照护问题：
{selected_problem_text}

检索到的具体照护建议 ProblemAdvice：
{problem_advice_text}

检索到的活动安排建议 ActivityAdvice：
{activity_advice_text}

检索到的环境调整建议 EnvironmentAdvice：
{environment_advice_text}

可引用的证据来源：
{evidence_source_text}

请基于以上信息，生成一个面向家属照护者的中文回答。

必须严格使用下面八个一级标题，标题文字不要改：

一、我理解的照护问题
二、可以先这样做
三、白天和夜间活动安排
四、环境调整建议
五、风险提醒
六、什么时候需要咨询专业人员
七、依据来源
八、说明：本回答用于照护教育支持，不能替代专业诊疗

写作要求：
- 语气温和、坚定、实用。
- 不要说“老人是故意的”“老人不讲理”。
- 可以解释行为可能与认知障碍、记忆下降、定向障碍或精神行为症状有关，但不要下诊断。
- “可以先这样做”部分请写成步骤化建议。
- “风险提醒”和“什么时候需要咨询专业人员”要清楚区分：前者提醒家属注意什么，后者说明何时需要找专业人员。
- 正文不要出现 [S1]、[P1]、[A1]、[E1] 这类引用编号。
- 正文不要列出具体指南、共识或标准名称。
- “依据来源”部分只写一句：本轮建议基于知识图谱检索到的照护建议和证据生成，具体依据可在回答下方“查看本轮依据与证据”中查看。
- 最后一节必须明确：本回答用于照护教育支持，不能替代医生、护士、记忆门诊或急诊判断。
"""


def append_missing_required_sections(answer_md: str) -> str:
    """
    Ensure every required section title appears at least once.
    If the LLM misses a section, append a conservative placeholder.
    """
    answer_md = answer_md.strip()

    missing = [section for section in REQUIRED_SECTIONS if section not in answer_md]

    if not missing:
        return answer_md

    answer_md += "\n\n---\n\n"

    for section in missing:
        answer_md += f"\n\n{section}\n\n"
        if section.startswith("七、"):
            answer_md += "本次回答依据前述检索到的照护建议和证据来源生成。\n"
        elif section.startswith("八、"):
            answer_md += "本回答仅用于照护教育支持，不能替代医生、护士、记忆门诊或急诊的专业判断。\n"
        else:
            answer_md += "本次检索到的相关内容较少，可结合前述建议谨慎处理，并在情况加重时咨询专业人员。\n"

    return answer_md.strip()


def add_safety_note_if_missing(answer_md: str) -> str:
    required_note = "不能替代"
    if required_note in answer_md and "照护教育支持" in answer_md:
        return answer_md

    answer_md += (
        "\n\n八、说明：本回答用于照护教育支持，不能替代专业诊疗\n\n"
        "本回答仅用于照护教育支持，不能替代医生、护士、记忆门诊或急诊的专业判断。"
    )

    return answer_md.strip()



def remove_inline_citation_markers(answer_md: str) -> str:
    """
    Remove machine-style citation markers such as [S1], [P2], [A3], [E4].
    The evidence remains available in the UI evidence panel.
    """
    answer_md = re.sub(r"(?:\[(?:S|P|A|E)\d+\])+", "", answer_md)
    answer_md = re.sub(r"（\s*）", "", answer_md)
    answer_md = re.sub(r"\s+，", "，", answer_md)
    answer_md = re.sub(r"\s+。", "。", answer_md)
    return answer_md.strip()


def rewrite_source_section_for_user(answer_md: str) -> str:
    """
    Keep the seventh section, but make it user-facing.
    Do not list references in the chat answer because the UI has an evidence panel.
    """
    replacement = (
        "七、依据来源\n\n"
        "本轮建议基于知识图谱检索到的照护建议和证据生成。"
        "如需查看具体依据、证据摘录和相关知识条目，可点击回答下方的“查看本轮依据与证据”。\n\n"
    )

    pattern = r"七、依据来源\s*.*?(?=\n\s*八、说明：本回答用于照护教育支持，不能替代专业诊疗)"

    if re.search(pattern, answer_md, flags=re.S):
        answer_md = re.sub(pattern, replacement.rstrip(), answer_md, flags=re.S)
    else:
        # If section 7 is missing, append_missing_required_sections will handle it later.
        pass

    return answer_md.strip()



def generate_answer(
    retrieval_result: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 3200,
    max_problem_advice: int = DEFAULT_MAX_PROBLEM_ADVICE,
    max_activity_advice: int = DEFAULT_MAX_ACTIVITY_ADVICE,
    max_environment_advice: int = DEFAULT_MAX_ENVIRONMENT_ADVICE,
    max_evidence_sources: int = DEFAULT_MAX_EVIDENCE_SOURCES,
) -> Dict[str, Any]:
    start = time.time()

    context = build_context_for_prompt(
        retrieval_result=retrieval_result,
        max_problem_advice=max_problem_advice,
        max_activity_advice=max_activity_advice,
        max_environment_advice=max_environment_advice,
        max_evidence_sources=max_evidence_sources,
    )

    DeepSeekClient = load_deepseek_client_class()
    client = DeepSeekClient()

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(context)

    try:
        answer_md = client.chat_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        success = True
        error = ""

    except Exception as e:
        success = False
        error = repr(e)
        answer_md = build_fallback_answer(context)

    answer_md = remove_inline_citation_markers(answer_md)
    answer_md = rewrite_source_section_for_user(answer_md)
    answer_md = append_missing_required_sections(answer_md)
    answer_md = add_safety_note_if_missing(answer_md)

    elapsed = round(time.time() - start, 3)

    result = {
        "success": success,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v02_final_workflow_step06",
        "question": retrieval_result.get("question"),
        "selected_problem_ids": retrieval_result.get("selected_problem_ids"),
        "selected_problems": retrieval_result.get("selected_problems"),
        "retrieval_counts": retrieval_result.get("retrieval_counts"),
        "generation_config": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_problem_advice": max_problem_advice,
            "max_activity_advice": max_activity_advice,
            "max_environment_advice": max_environment_advice,
            "max_evidence_sources": max_evidence_sources,
        },
        "answer_md": answer_md,
        "prompt_context": context,
        "error": error,
        "elapsed_seconds": elapsed,
    }

    return result


def build_fallback_answer(context: Dict[str, Any]) -> str:
    """
    Conservative fallback if DeepSeek request fails.
    """
    question = context.get("question", "")

    selected = context.get("selected_problem_lines") or []
    p_advices = context.get("problem_advice_lines") or []
    a_advices = context.get("activity_advice_lines") or []
    e_advices = context.get("environment_advice_lines") or []
    sources = context.get("evidence_source_lines") or []

    def bullet_from_lines(lines: List[str], limit: int = 5) -> str:
        if not lines:
            return "- 本次检索到的相关内容较少，可先记录表现并咨询专业人员。"
        result_lines = []
        for line in lines[:limit]:
            # Extract advice after " | 建议: "
            advice = line
            if " | 建议: " in line:
                advice = line.split(" | 建议: ", 1)[1]
            if " | 来源: " in advice:
                advice = advice.split(" | 来源: ", 1)[0]
            result_lines.append(f"- {advice}")
        return "\n".join(result_lines)

    selected_text = "\n".join(selected) if selected else "- 暂未识别到明确问题。"

    source_text = "\n".join(f"- {line}" for line in sources[:6]) if sources else "- 本次检索到的来源信息较少。"

    return f"""
一、我理解的照护问题

您描述的问题是：{question}

系统识别到的相关照护问题包括：

{selected_text}

二、可以先这样做

{bullet_from_lines(p_advices, limit=6)}

三、白天和夜间活动安排

{bullet_from_lines(a_advices, limit=5)}

四、环境调整建议

{bullet_from_lines(e_advices, limit=5)}

五、风险提醒

- 注意观察老人是否出现走失、跌倒、攻击行为、自伤、误服、误吸、噎食、严重呛咳、持续幻觉或妄想等风险。
- 如果老人夜间活动增多，家属应重点检查地面障碍物、照明、门窗和危险物品，避免夜间发生意外。
- 如果老人情绪激动，尽量避免争辩、责备或强行纠正。

六、什么时候需要咨询专业人员

- 如果相关行为持续加重，或安抚后仍无法改善，建议咨询记忆门诊、神经内科、精神心理科或专业护理人员。
- 如果出现明显攻击、自伤、伤人、误服、严重呛咳、噎食、跌倒后异常、意识改变等情况，应及时就医或寻求急诊帮助。

七、依据来源

{source_text}

八、说明：本回答用于照护教育支持，不能替代专业诊疗

本回答仅用于照护教育支持，不能替代医生、护士、记忆门诊或急诊的专业判断。若老人情况变化明显或出现安全风险，请及时联系专业人员。
""".strip()


def write_answer_result(result: Dict[str, Any]) -> Dict[str, str]:
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    timestamp_md_path = ANSWERS_DIR / f"answer_{timestamp}.md"
    timestamp_json_path = ANSWERS_DIR / f"answer_result_{timestamp}.json"

    answer_md = result["answer_md"]

    LAST_ANSWER_MD_PATH.write_text(answer_md, encoding="utf-8")
    timestamp_md_path.write_text(answer_md, encoding="utf-8")

    json_text = json.dumps(result, ensure_ascii=False, indent=2)
    LAST_ANSWER_JSON_PATH.write_text(json_text, encoding="utf-8")
    timestamp_json_path.write_text(json_text, encoding="utf-8")

    return {
        "last_answer_md": str(LAST_ANSWER_MD_PATH),
        "last_answer_json": str(LAST_ANSWER_JSON_PATH),
        "timestamp_answer_md": str(timestamp_md_path),
        "timestamp_answer_json": str(timestamp_json_path),
    }


def print_answer_preview(result: Dict[str, Any]) -> None:
    print_header("Answer preview")
    print(result["answer_md"])

    print_header("Generation summary")
    print(f"Success: {result.get('success')}")
    print(f"Elapsed seconds: {result.get('elapsed_seconds')}")
    print(f"Question: {result.get('question')}")
    print(f"Selected problem IDs: {result.get('selected_problem_ids')}")
    if result.get("error"):
        print(f"Error: {result.get('error')}")


def build_retrieval_result_from_question(
    question: str,
    problem_ids: Optional[List[str]],
    no_auto_map: bool,
    no_debug: bool,
) -> Dict[str, Any]:
    retrieve_func, write_retrieval_result = load_retrieve_function()

    result = retrieve_func(
        question=question,
        problem_ids=normalize_problem_ids(problem_ids),
        auto_map=not no_auto_map,
        include_debug=not no_debug,
    )

    write_retrieval_result(result)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 06: Generate evidence-grounded caregiver answer."
    )

    parser.add_argument(
        "--retrieval_result",
        type=str,
        default="",
        help="Path to retrieval result JSON. Default: outputs/retrieval_debug/last_retrieval_result.json",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Optional new question. If provided, Step 05 retrieval will be called first.",
    )

    parser.add_argument(
        "--problem_ids",
        nargs="*",
        default=None,
        help="Optional Problem IDs when --question is provided, e.g. --problem_ids F3 H5",
    )

    parser.add_argument(
        "--no_auto_map",
        action="store_true",
        help="When --question is provided and no problem_ids are given, disable auto mapping.",
    )

    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="When calling Step 05 internally, do not include retrieval debug details.",
    )

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=3200)

    parser.add_argument("--max_problem_advice", type=int, default=DEFAULT_MAX_PROBLEM_ADVICE)
    parser.add_argument("--max_activity_advice", type=int, default=DEFAULT_MAX_ACTIVITY_ADVICE)
    parser.add_argument("--max_environment_advice", type=int, default=DEFAULT_MAX_ENVIRONMENT_ADVICE)
    parser.add_argument("--max_evidence_sources", type=int, default=DEFAULT_MAX_EVIDENCE_SOURCES)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header("Step 06: Generate evidence-grounded care answer")
    print("[READ-ONLY] This script reads retrieval context and calls DeepSeek. It does not modify KG.")

    if args.question:
        print(f"Question provided. Step 05 retrieval will run first: {args.question}")
        if args.problem_ids:
            print(f"Problem IDs: {args.problem_ids}")
        elif not args.no_auto_map:
            print("Problem IDs not provided. Auto-mapping will be used.")
        else:
            print("Problem IDs not provided and auto-mapping disabled.")

        retrieval_result = build_retrieval_result_from_question(
            question=args.question,
            problem_ids=args.problem_ids,
            no_auto_map=args.no_auto_map,
            no_debug=args.no_debug,
        )

    else:
        retrieval_path = Path(args.retrieval_result) if args.retrieval_result else DEFAULT_RETRIEVAL_RESULT_PATH
        if not retrieval_path.is_absolute():
            retrieval_path = PROJECT_ROOT / retrieval_path

        print(f"Using retrieval result: {retrieval_path}")
        retrieval_result = load_json(retrieval_path)

    result = generate_answer(
        retrieval_result=retrieval_result,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_problem_advice=args.max_problem_advice,
        max_activity_advice=args.max_activity_advice,
        max_environment_advice=args.max_environment_advice,
        max_evidence_sources=args.max_evidence_sources,
    )

    paths = write_answer_result(result)

    print_answer_preview(result)

    print_header("Final result")
    print("[PASS] Step 06 answer generation finished.")
    print(f"[OK] Last answer markdown: {paths['last_answer_md']}")
    print(f"[OK] Last answer JSON: {paths['last_answer_json']}")
    print(f"[OK] Timestamp answer markdown: {paths['timestamp_answer_md']}")
    print(f"[OK] Timestamp answer JSON: {paths['timestamp_answer_json']}")
    print("Next step: write a unified agent script that runs mapping + retrieval + answer generation in one call.")


if __name__ == "__main__":
    main()
