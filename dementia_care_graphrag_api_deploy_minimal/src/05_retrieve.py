"""
Step 05: Retrieve evidence context by GraphRAG.

This module performs:
1. Optional problem mapping by calling src/04_problem_mapper.py
2. Neo4j graph retrieval by selected Problem IDs
3. Corpus graph fallback retrieval by related_problem_ids
4. FAISS Advice vector retrieval
5. Keyword retrieval
6. Fusion ranking
7. Final context grouping for answer generation

Inputs:
1. question
2. optional problem_ids
3. local Neo4j v02_final
4. outputs/indexes/advice_bgem3_flatip.faiss
5. outputs/indexes/advice_bgem3_meta.jsonl
6. outputs/indexes/problem_bgem3_flatip.faiss indirectly through 04_problem_mapper.py if auto mapping is used

Outputs:
1. outputs/retrieval_debug/last_retrieval_result.json
2. outputs/retrieval_debug/retrieval_result_<timestamp>.json

This script does NOT modify Neo4j.
It only reads local Neo4j and local FAISS/corpus files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
RETRIEVAL_DEBUG_DIR = PROJECT_ROOT / "outputs" / "retrieval_debug"

ADVICE_FAISS_PATH = INDEX_DIR / "advice_bgem3_flatip.faiss"
ADVICE_META_PATH = INDEX_DIR / "advice_bgem3_meta.jsonl"

LAST_RETRIEVAL_PATH = RETRIEVAL_DEBUG_DIR / "last_retrieval_result.json"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"

DEFAULT_TOP_ADVICE_VECTOR = 30
DEFAULT_TOP_KEYWORD = 30

DEFAULT_MAX_PROBLEM_ADVICE = 12
DEFAULT_MAX_ACTIVITY_ADVICE = 6
DEFAULT_MAX_ENVIRONMENT_ADVICE = 6

ADVICE_LABELS = {"ProblemAdvice", "ActivityAdvice", "EnvironmentAdvice"}

TYPE_TO_CONTEXT_KEY = {
    "ProblemAdvice": "problem_advice",
    "ActivityAdvice": "activity_advice",
    "EnvironmentAdvice": "environment_advice",
}

DOMAIN_KEYWORDS = [
    "晚上", "夜间", "白天", "睡", "睡觉", "入睡", "起床", "昼夜颠倒",
    "走", "徘徊", "游荡", "来回走", "外跑", "走失",
    "偷", "被偷", "被害", "怀疑", "妄想", "幻觉", "有人",
    "洗澡", "梳洗", "刷牙", "拒绝", "抗拒", "攻击", "打", "踢", "推", "咬",
    "吃饭", "进食", "餐具", "吞咽", "咀嚼", "呛", "呛咳", "噎", "噎食", "误吸",
    "厕所", "大小便", "失禁", "便秘", "排泄",
    "穿衣", "脱衣", "衣服",
    "跌倒", "坠床", "压疮", "感染", "肺炎", "危险",
    "淡漠", "重复", "尖叫", "哭", "笑", "骂人", "破坏",
]

RISK_KEYWORDS = [
    "跌倒", "坠床", "自伤", "伤人", "外跑", "走失", "误吸", "误服",
    "呛咳", "噎食", "肺炎", "感染", "压疮", "攻击", "打人", "咬人",
    "意识", "急性", "发热", "窒息", "危险",
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
    RETRIEVAL_DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def load_env() -> Dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_LOCAL_PASSWORD", "")

    embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if not neo4j_password:
        raise RuntimeError("NEO4J_LOCAL_PASSWORD is empty. Please set it in .env.")

    return {
        "neo4j_uri": neo4j_uri,
        "neo4j_user": neo4j_user,
        "neo4j_password": neo4j_password,
        "embedding_model": embedding_model,
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    docs: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path}, line {line_no}: {e}") from e

    return docs


def import_faiss():
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss is not installed.\n"
            "Please run:\n"
            "  conda install -c conda-forge faiss-cpu -y"
        ) from e
    return faiss


def load_embedding_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Please run: pip install sentence-transformers"
        ) from e

    return SentenceTransformer(model_name)



@lru_cache(maxsize=1)
def get_advice_meta_cached():
    """
    Cache advice metadata in the current Python process.
    """
    return load_jsonl(ADVICE_META_PATH)


@lru_cache(maxsize=2)
def get_embedding_model_cached(model_name: str):
    """
    Cache BGE-M3 embedding model in the current Python process.
    This avoids reloading model weights on every Streamlit turn.
    """
    return load_embedding_model(model_name)


@lru_cache(maxsize=1)
def get_faiss_index_cached(index_path_str: str):
    """
    Cache FAISS index in the current Python process.
    """
    faiss = import_faiss()
    return faiss.read_index(index_path_str)


def query_instruction(query: str) -> str:
    query = clean_text(query)
    return f"为这个句子生成表示以用于检索相关照护知识：{query}"


def embed_query(model, query: str) -> np.ndarray:
    q_emb = model.encode(
        [query_instruction(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return q_emb


def load_problem_mapper_class():
    module_path = SRC_DIR / "04_problem_mapper.py"

    if not module_path.exists():
        raise FileNotFoundError(
            f"Cannot find {module_path}. Please create src/04_problem_mapper.py first."
        )

    spec = importlib.util.spec_from_file_location("step04_problem_mapper", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.LLMProblemMapper



@lru_cache(maxsize=1)
def get_problem_mapper_cached():
    """
    Cache LLMProblemMapper in the current Python process.

    This avoids reloading 04_problem_mapper.py, Problem FAISS index,
    and BGE-M3 problem embedding model on every Streamlit turn.
    """
    MapperClass = load_problem_mapper_class()
    # Speed mode:
    # Problem catalog has only 58 items, so fallback LLM mapping can use the full catalog directly.
    # This avoids loading the separate Problem BGE-M3 / Problem FAISS assets in 04_problem_mapper.py.
    return MapperClass(enable_vector_candidates=False)


def normalize_problem_ids(problem_ids: Optional[List[str]]) -> List[str]:
    if not problem_ids:
        return []

    cleaned: List[str] = []
    seen = set()

    for pid in problem_ids:
        pid = clean_text(pid).upper()
        if not pid:
            continue
        if pid not in seen:
            cleaned.append(pid)
            seen.add(pid)

    return cleaned



def fast_rule_map_problem_ids(question: str) -> Dict[str, Any]:
    """
    Fast local problem mapping for common dementia-care questions.

    Goal:
    - Avoid DeepSeek problem mapping for high-confidence common cases.
    - Fall back to LLMProblemMapper when rules are uncertain.

    This does not replace GraphRAG retrieval.
    It only decides the initial Problem IDs faster.
    """
    q = clean_text(question)
    q_compact = re.sub(r"\s+", "", q)

    matched: List[Dict[str, Any]] = []

    def add(pid: str, reason: str, confidence: float = 0.92):
        if pid not in [x["id"] for x in matched]:
            matched.append(
                {
                    "id": pid,
                    "confidence": confidence,
                    "reason": reason,
                }
            )

    # H14 妄想 / 被窃妄想 / 被害妄想
    if re.search(r"(偷|被偷|被窃|钱.*丢|东西.*丢|找不到钱|钱没有丢|解释钱|有人害|被害|怀疑.*外遇)", q_compact):
        add("H14", "命中被窃妄想/被害妄想相关关键词", 0.95)

    # F2 夜间多次起床
    if re.search(r"(晚上|夜里|半夜|夜间).*(起来|起床|醒来|醒|下床|睡不着|不睡)", q_compact):
        add("F2", "命中夜间醒来/起床相关关键词", 0.93)

    # F3 昼夜颠倒
    if re.search(r"(白天睡|白天.*睡很多|晚上.*精神|昼夜颠倒|日夜颠倒|黑白颠倒)", q_compact):
        add("F3", "命中昼夜节律紊乱相关关键词", 0.93)

    # H5 徘徊和游荡
    if re.search(r"(走来走去|不停走|一直走|徘徊|游荡|来回走)", q_compact):
        add("H5", "命中徘徊/游荡相关关键词", 0.92)

    # I5 外跑
    if re.search(r"(往外跑|外跑|跑出去|自己出去|出门|开门出去|走到门口|门口|要出去|想出去)", q_compact):
        add("I5", "命中外跑/自行外出风险相关关键词", 0.94)

    # I6 走失
    if re.search(r"(走失|迷路|找不到家|回不了家|不知道怎么回家|找不到路)", q_compact):
        add("I6", "命中走失/迷路相关关键词", 0.94)

    # I9 呛咳
    if re.search(r"(喝水.*呛|喝水.*咳|吃饭.*呛|呛咳|老是呛|总是呛)", q_compact):
        add("I9", "命中呛咳相关关键词", 0.94)

    # I10 噎食
    if re.search(r"(噎食|噎住|卡住|吃饭.*卡|食物.*卡|吞不下)", q_compact):
        add("I10", "命中噎食/卡住相关关键词", 0.94)

    # D1 拒绝洗澡
    if re.search(r"(不洗澡|拒绝洗澡|不愿意洗澡|害怕洗澡|洗澡.*不愿意)", q_compact):
        add("D1", "命中拒绝洗澡相关关键词", 0.93)

    # D4 他人协助洗澡时攻击行为
    if re.search(r"(洗澡.*打人|洗澡.*攻击|洗澡.*骂人|帮.*洗澡.*打|协助洗澡.*打)", q_compact):
        add("D4", "命中协助洗澡时攻击行为相关关键词", 0.93)

    # A / eating refusal, simple additions for common future questions
    if re.search(r"(不吃饭|拒绝吃饭|不愿意吃饭|吃得很少|食欲差)", q_compact):
        add("A1", "命中拒绝进食/进食减少相关关键词", 0.88)

    # Keep top 4 to avoid overly broad retrieval.
    matched = matched[:4]
    matched_ids = [x["id"] for x in matched]

    return {
        "method": "fast_rule",
        "matched_problem_ids": matched_ids,
        "matched_problems": matched,
        "needs_clarification": False,
        "follow_up_question": "",
        "raw_question": question,
        "used_llm": False,
    }


def map_question_if_needed(
    question: str,
    problem_ids: Optional[List[str]],
    auto_map: bool,
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    explicit_ids = normalize_problem_ids(problem_ids)

    if explicit_ids:
        return explicit_ids, None

    if not auto_map:
        return [], None

    # Fast local mapping first.
    # For common high-confidence cases, skip DeepSeek problem mapping.
    fast_mapping_result = fast_rule_map_problem_ids(question)
    fast_ids = normalize_problem_ids(fast_mapping_result.get("matched_problem_ids") or [])

    if fast_ids:
        return fast_ids, fast_mapping_result

    # Fallback to DeepSeek-based mapper when local rules are uncertain.
    mapper = get_problem_mapper_cached()
    mapping_result = mapper.map_question(question)

    mapped_ids = mapping_result.get("matched_problem_ids") or [
        item.get("id")
        for item in mapping_result.get("matched_problems", [])
        if item.get("id")
    ]

    mapped_ids = normalize_problem_ids(mapped_ids)

    return mapped_ids, mapping_result


def connect_neo4j(env: Dict[str, Any]):
    try:
        driver = GraphDatabase.driver(
            env["neo4j_uri"],
            auth=(env["neo4j_user"], env["neo4j_password"]),
        )
        with driver.session() as session:
            session.run("RETURN 1 AS ok").single()
        return driver

    except AuthError as e:
        raise RuntimeError("Neo4j authentication failed. Please check .env.") from e
    except ServiceUnavailable as e:
        raise RuntimeError("Neo4j service unavailable. Please check local Neo4j.") from e


def run_query(driver, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    with driver.session() as session:
        result = session.run(query, params or {})
        return [record.data() for record in result]


def get_primary_advice_label(labels: List[str]) -> str:
    for label in ["ProblemAdvice", "ActivityAdvice", "EnvironmentAdvice"]:
        if label in labels:
            return label
    return labels[0] if labels else "Unknown"


def retrieve_graph_from_neo4j(
    driver,
    problem_ids: List[str],
    limit: int = 300,
) -> List[Dict[str, Any]]:
    """
    Retrieve Advice nodes connected to selected Problems through 1-2 hops.

    This is mainly for path explanation.
    Corpus graph fallback is still used because Step 01 already normalized
    relation/property/id-prefix evidence into related_problem_ids.
    """
    if not problem_ids:
        return []

    rows = run_query(
        driver,
        """
        MATCH (p:Problem)
        WHERE p.id IN $problem_ids
        MATCH path = (p)-[*1..2]-(a)
        WHERE a:ProblemAdvice OR a:ActivityAdvice OR a:EnvironmentAdvice
        RETURN DISTINCT
            p.id AS problem_id,
            p.name AS problem_name,
            labels(a) AS advice_labels,
            a.id AS advice_id,
            a.name AS advice_name,
            a.content_text AS advice_text,
            elementId(a) AS advice_element_id,
            properties(a) AS advice_properties,
            [rel IN relationships(path) | type(rel)] AS rel_path,
            [n IN nodes(path) | {
                labels: labels(n),
                id: n.id,
                name: n.name
            }] AS node_path
        ORDER BY problem_id, advice_id
        LIMIT $limit
        """,
        {
            "problem_ids": problem_ids,
            "limit": limit,
        },
    )

    results: List[Dict[str, Any]] = []

    for row in rows:
        labels = row.get("advice_labels") or []
        node_label = get_primary_advice_label(labels)
        advice_id = clean_text(row.get("advice_id"))

        if not advice_id or node_label not in ADVICE_LABELS:
            continue

        doc_id = f"{node_label}:{advice_id}"
        props = row.get("advice_properties") or {}

        advice_text = (
            clean_text(row.get("advice_text"))
            or clean_text(props.get("content_text"))
            or clean_text(props.get("name"))
        )

        results.append(
            {
                "doc_id": doc_id,
                "node_label": node_label,
                "advice_id": advice_id,
                "advice_name": clean_text(row.get("advice_name")),
                "advice_text": advice_text,
                "matched_problem_id": clean_text(row.get("problem_id")),
                "matched_problem_name": clean_text(row.get("problem_name")),
                "rel_path": row.get("rel_path") or [],
                "node_path": row.get("node_path") or [],
                "retrieval_source": "graph_neo4j",
            }
        )

    return results


def retrieve_graph_from_corpus(
    advice_meta: List[Dict[str, Any]],
    problem_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Stable graph fallback based on Step 01 exported related_problem_ids.
    """
    if not problem_ids:
        return []

    problem_set = set(problem_ids)
    results: List[Dict[str, Any]] = []

    for doc in advice_meta:
        related_ids = doc.get("related_problem_ids") or []
        matched = [pid for pid in related_ids if pid in problem_set]

        if not matched:
            continue

        item = dict(doc)
        item["matched_problem_ids"] = matched
        item["retrieval_source"] = "graph_corpus"
        results.append(item)

    return results


def search_faiss_advice(
    model,
    advice_meta: List[Dict[str, Any]],
    question: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    if not ADVICE_FAISS_PATH.exists():
        raise FileNotFoundError(
            f"Advice FAISS index not found: {ADVICE_FAISS_PATH}\n"
            "Please run: python src/02_build_vector_index.py"
        )

    index = get_faiss_index_cached(str(ADVICE_FAISS_PATH))

    top_k = min(top_k, len(advice_meta))
    q_emb = embed_query(model, question)

    scores, indices = index.search(q_emb, top_k)

    results: List[Dict[str, Any]] = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0 or idx >= len(advice_meta):
            continue

        doc = dict(advice_meta[int(idx)])
        doc["retrieval_source"] = "vector_advice"
        doc["vector_score"] = float(score)
        doc["vector_rank"] = rank
        results.append(doc)

    return results


def selected_problem_objects_from_mapping(
    selected_problem_ids: List[str],
    mapping_result: Optional[Dict[str, Any]],
    advice_meta: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Build selected problem objects for final context.

    Prefer mapping_result.
    If retrieval is called with explicit problem IDs and no mapping_result,
    infer names from advice_meta related_problem_details.
    """
    by_id: Dict[str, Dict[str, Any]] = {}

    if mapping_result:
        for item in mapping_result.get("matched_problems", []) or []:
            pid = clean_text(item.get("id")).upper()
            if not pid:
                continue
            by_id[pid] = {
                "id": pid,
                "name": clean_text(item.get("name") or item.get("content")),
                "content": clean_text(item.get("content") or item.get("name")),
                "category": clean_text(item.get("category")),
                "confidence": item.get("confidence"),
                "evidence_in_question": clean_text(item.get("evidence_in_question")),
                "reason": clean_text(item.get("reason")),
            }

    # Fallback from corpus metadata.
    for doc in advice_meta:
        for detail in doc.get("related_problem_details") or []:
            pid = clean_text(detail.get("id")).upper()
            if not pid or pid in by_id:
                continue
            by_id[pid] = {
                "id": pid,
                "name": clean_text(detail.get("name")),
                "content": clean_text(detail.get("name")),
                "category": clean_text(detail.get("category")),
            }

        related_ids = doc.get("related_problem_ids") or []
        related_names = doc.get("related_problem_names") or []
        for i, pid in enumerate(related_ids):
            pid = clean_text(pid).upper()
            if not pid or pid in by_id:
                continue
            name = clean_text(related_names[i] if i < len(related_names) else "")
            by_id[pid] = {
                "id": pid,
                "name": name,
                "content": name,
                "category": "",
            }

    selected: List[Dict[str, Any]] = []
    for pid in selected_problem_ids:
        selected.append(by_id.get(pid, {"id": pid, "name": "", "content": "", "category": ""}))

    return selected


def extract_keywords(
    question: str,
    selected_problems: List[Dict[str, Any]],
) -> List[str]:
    keywords: List[str] = []
    question = clean_text(question)

    for kw in DOMAIN_KEYWORDS:
        if kw in question and kw not in keywords:
            keywords.append(kw)

    for problem in selected_problems:
        pid = clean_text(problem.get("id"))
        name = clean_text(problem.get("name") or problem.get("content"))

        if pid and pid not in keywords:
            keywords.append(pid)

        if name and name not in keywords:
            keywords.append(name)

        fragments = re.split(r"[，,、；;：:\-\s（）()]+", name)
        for frag in fragments:
            frag = clean_text(frag)
            if len(frag) >= 2 and frag not in keywords:
                keywords.append(frag)

    return keywords


def keyword_score_doc(doc: Dict[str, Any], keywords: List[str]) -> Tuple[float, List[str]]:
    if not keywords:
        return 0.0, []

    fields = [
        doc.get("doc_id"),
        doc.get("node_label"),
        doc.get("advice_id"),
        doc.get("advice_name"),
        doc.get("advice_text"),
        " ".join(doc.get("related_problem_ids") or []),
        " ".join(doc.get("related_problem_names") or []),
        doc.get("condition_rule_text"),
        doc.get("source_title"),
        doc.get("source_type"),
        doc.get("year"),
        doc.get("heading_path"),
        doc.get("evidence_text"),
        doc.get("search_text"),
    ]

    haystack = "\n".join(clean_text(x) for x in fields if clean_text(x))

    matched: List[str] = []
    score = 0.0

    for kw in keywords:
        kw = clean_text(kw)
        if not kw:
            continue

        if kw in haystack:
            matched.append(kw)

            if len(kw) >= 6:
                score += 0.25
            elif len(kw) >= 3:
                score += 0.15
            else:
                score += 0.08

    return min(score, 0.8), matched


def retrieve_keyword_advice(
    advice_meta: List[Dict[str, Any]],
    keywords: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []

    for doc in advice_meta:
        score, matched = keyword_score_doc(doc, keywords)
        if score <= 0:
            continue

        item = dict(doc)
        item["retrieval_source"] = "keyword"
        item["keyword_score"] = score
        item["matched_keywords"] = matched
        scored.append(item)

    scored.sort(key=lambda x: x.get("keyword_score", 0.0), reverse=True)
    return scored[:top_k]


def build_doc_map(advice_meta: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    doc_map: Dict[str, Dict[str, Any]] = {}

    for doc in advice_meta:
        doc_id = clean_text(doc.get("doc_id"))
        if doc_id:
            doc_map[doc_id] = doc

    return doc_map


def add_source(item: Dict[str, Any], source: str) -> None:
    sources = item.setdefault("retrieval_sources", [])
    if source not in sources:
        sources.append(source)


def merge_unique(existing: Optional[List[Any]], new_items: Optional[List[Any]]) -> List[Any]:
    result: List[Any] = list(existing or [])

    for value in new_items or []:
        if value is None:
            continue
        if value not in result:
            result.append(value)

    return result


def fuse_results(
    advice_meta: List[Dict[str, Any]],
    graph_neo4j_results: List[Dict[str, Any]],
    graph_corpus_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    keyword_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Fusion strategy:
    - graph_neo4j: strongest source, path-explainable
    - graph_corpus: stable graph fallback from Step 01
    - vector: semantic supplement
    - keyword: lexical supplement
    - evidence/source/update_id bonuses preserve v02 evidence-aware knowledge
    """
    doc_map = build_doc_map(advice_meta)
    fused: Dict[str, Dict[str, Any]] = {}

    def get_or_create(doc_id: str) -> Dict[str, Any]:
        if doc_id not in fused:
            base = dict(doc_map.get(doc_id, {"doc_id": doc_id}))
            base["retrieval_sources"] = []
            base["matched_problem_ids"] = list(base.get("matched_problem_ids") or [])
            base["matched_problem_names"] = list(base.get("matched_problem_names") or [])
            base["vector_score"] = 0.0
            base["keyword_score"] = 0.0
            base["graph_score"] = 0.0
            base["graph_paths"] = []
            base["matched_keywords"] = []
            fused[doc_id] = base
        return fused[doc_id]

    # 1. Neo4j graph channel.
    for result in graph_neo4j_results:
        doc_id = clean_text(result.get("doc_id"))
        if not doc_id:
            continue

        item = get_or_create(doc_id)
        add_source(item, "graph_neo4j")
        item["graph_score"] = max(float(item.get("graph_score", 0.0)), 2.5)

        pid = clean_text(result.get("matched_problem_id")).upper()
        pname = clean_text(result.get("matched_problem_name"))

        item["matched_problem_ids"] = merge_unique(item.get("matched_problem_ids"), [pid])
        item["matched_problem_names"] = merge_unique(item.get("matched_problem_names"), [pname])

        item["graph_paths"].append(
            {
                "matched_problem_id": pid,
                "matched_problem_name": pname,
                "rel_path": result.get("rel_path") or [],
                "node_path": result.get("node_path") or [],
            }
        )

    # 2. Corpus graph channel.
    for result in graph_corpus_results:
        doc_id = clean_text(result.get("doc_id"))
        if not doc_id:
            continue

        item = get_or_create(doc_id)
        add_source(item, "graph_corpus")
        item["graph_score"] = max(float(item.get("graph_score", 0.0)), 2.0)
        item["matched_problem_ids"] = merge_unique(
            item.get("matched_problem_ids"),
            result.get("matched_problem_ids") or [],
        )

    # 3. Vector channel.
    for result in vector_results:
        doc_id = clean_text(result.get("doc_id"))
        if not doc_id:
            continue

        item = get_or_create(doc_id)
        add_source(item, "vector")
        item["vector_score"] = max(
            float(item.get("vector_score", 0.0)),
            float(result.get("vector_score", 0.0)),
        )
        item["vector_rank"] = min(
            int(item.get("vector_rank", 999999)),
            int(result.get("vector_rank", 999999)),
        )

    # 4. Keyword channel.
    for result in keyword_results:
        doc_id = clean_text(result.get("doc_id"))
        if not doc_id:
            continue

        item = get_or_create(doc_id)
        add_source(item, "keyword")
        item["keyword_score"] = max(
            float(item.get("keyword_score", 0.0)),
            float(result.get("keyword_score", 0.0)),
        )
        item["matched_keywords"] = merge_unique(
            item.get("matched_keywords"),
            result.get("matched_keywords") or [],
        )

    final_items: List[Dict[str, Any]] = []

    for item in fused.values():
        evidence_bonus = 0.20 if clean_text(item.get("evidence_text")) else 0.0
        source_bonus = 0.10 if clean_text(item.get("source_title")) else 0.0
        v02_bonus = 0.15 if clean_text(item.get("update_id")) else 0.0

        risk_text = " ".join(
            [
                clean_text(item.get("advice_text")),
                clean_text(item.get("advice_name")),
                " ".join(item.get("related_problem_names") or []),
            ]
        )
        risk_bonus = 0.05 if any(kw in risk_text for kw in RISK_KEYWORDS) else 0.0

        graph_score = float(item.get("graph_score", 0.0))
        vector_score = float(item.get("vector_score", 0.0))
        keyword_score = float(item.get("keyword_score", 0.0))

        final_score = (
            graph_score
            + vector_score
            + keyword_score
            + evidence_bonus
            + source_bonus
            + v02_bonus
            + risk_bonus
        )

        item["evidence_bonus"] = evidence_bonus
        item["source_bonus"] = source_bonus
        item["v02_bonus"] = v02_bonus
        item["risk_bonus"] = risk_bonus
        item["final_score"] = round(final_score, 6)

        final_items.append(item)

    final_items.sort(
        key=lambda x: (
            float(x.get("final_score", 0.0)),
            float(x.get("graph_score", 0.0)),
            float(x.get("vector_score", 0.0)),
            float(x.get("keyword_score", 0.0)),
        ),
        reverse=True,
    )

    return final_items


def compact_advice_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": item.get("doc_id"),
        "node_label": item.get("node_label"),
        "advice_id": item.get("advice_id"),
        "advice_name": item.get("advice_name"),
        "advice_text": item.get("advice_text"),
        "related_problem_ids": item.get("related_problem_ids"),
        "related_problem_names": item.get("related_problem_names"),
        "matched_problem_ids": item.get("matched_problem_ids"),
        "matched_problem_names": item.get("matched_problem_names"),
        "retrieval_sources": item.get("retrieval_sources"),
        "final_score": item.get("final_score"),
        "graph_score": item.get("graph_score"),
        "vector_score": item.get("vector_score"),
        "keyword_score": item.get("keyword_score"),
        "matched_keywords": item.get("matched_keywords"),
        "source_title": item.get("source_title"),
        "source_type": item.get("source_type"),
        "year": item.get("year"),
        "heading_path": item.get("heading_path"),
        "evidence_text": item.get("evidence_text"),
        "chunk_id": item.get("chunk_id"),
        "update_id": item.get("update_id"),
        "condition_rule_text": item.get("condition_rule_text"),
        "graph_paths": item.get("graph_paths"),
    }


def build_evidence_sources(
    fused_results: List[Dict[str, Any]],
    max_sources: int = 12,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()

    for item in fused_results:
        source_title = clean_text(item.get("source_title"))
        heading_path = clean_text(item.get("heading_path"))
        evidence_text = clean_text(item.get("evidence_text"))
        chunk_id = clean_text(item.get("chunk_id"))

        if not source_title and not evidence_text:
            continue

        key = (source_title, heading_path, chunk_id, evidence_text[:100])
        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source_title": source_title,
                "source_type": clean_text(item.get("source_type")),
                "year": clean_text(item.get("year")),
                "heading_path": heading_path,
                "chunk_id": chunk_id,
                "evidence_text": evidence_text,
                "doc_id": item.get("doc_id"),
                "advice_id": item.get("advice_id"),
            }
        )

        if len(sources) >= max_sources:
            break

    return sources


def group_final_context(
    fused_results: List[Dict[str, Any]],
    selected_problems: List[Dict[str, Any]],
    max_problem_advice: int,
    max_activity_advice: int,
    max_environment_advice: int,
) -> Dict[str, Any]:
    context = {
        "selected_problems": selected_problems,
        "problem_advice": [],
        "activity_advice": [],
        "environment_advice": [],
        "evidence_sources": [],
    }

    limits = {
        "ProblemAdvice": max_problem_advice,
        "ActivityAdvice": max_activity_advice,
        "EnvironmentAdvice": max_environment_advice,
    }

    for item in fused_results:
        label = item.get("node_label")
        key = TYPE_TO_CONTEXT_KEY.get(label)

        if not key:
            continue

        if len(context[key]) >= limits.get(label, 0):
            continue

        context[key].append(compact_advice_item(item))

    context["evidence_sources"] = build_evidence_sources(fused_results)

    return context


def retrieve(
    question: str,
    problem_ids: Optional[List[str]] = None,
    auto_map: bool = True,
    top_advice_vector: int = DEFAULT_TOP_ADVICE_VECTOR,
    top_keyword: int = DEFAULT_TOP_KEYWORD,
    max_problem_advice: int = DEFAULT_MAX_PROBLEM_ADVICE,
    max_activity_advice: int = DEFAULT_MAX_ACTIVITY_ADVICE,
    max_environment_advice: int = DEFAULT_MAX_ENVIRONMENT_ADVICE,
    include_debug: bool = True,
) -> Dict[str, Any]:
    start = time.time()
    ensure_dirs()

    question = clean_text(question)
    if not question:
        raise ValueError("question is empty.")

    env = load_env()

    advice_meta = get_advice_meta_cached()

    selected_problem_ids, mapping_result = map_question_if_needed(
        question=question,
        problem_ids=problem_ids,
        auto_map=auto_map,
    )

    selected_problems = selected_problem_objects_from_mapping(
        selected_problem_ids=selected_problem_ids,
        mapping_result=mapping_result,
        advice_meta=advice_meta,
    )

    # Lazy-load BGE-M3 only when vector retrieval is really needed.
    # For common mapped Problem IDs, graph retrieval is usually enough.
    model = None

    graph_neo4j_results: List[Dict[str, Any]] = []
    driver = None

    try:
        driver = connect_neo4j(env)
        graph_neo4j_results = retrieve_graph_from_neo4j(
            driver=driver,
            problem_ids=selected_problem_ids,
            limit=300,
        )
    finally:
        if driver is not None:
            driver.close()

    graph_corpus_results = retrieve_graph_from_corpus(
        advice_meta=advice_meta,
        problem_ids=selected_problem_ids,
    )

    # Speed mode:
    # If graph retrieval already found enough evidence, skip BGE-M3 vector retrieval.
    # This avoids loading / running BGE-M3 for common well-covered problems.
    graph_total_before_vector = len(graph_neo4j_results) + len(graph_corpus_results)
    vector_skipped = False

    if selected_problem_ids and graph_total_before_vector >= 8:
        vector_results = []
        vector_skipped = True
    else:
        model = get_embedding_model_cached(env["embedding_model"])
        vector_results = search_faiss_advice(
            model=model,
            advice_meta=advice_meta,
            question=question,
            top_k=top_advice_vector,
        )

    keywords = extract_keywords(
        question=question,
        selected_problems=selected_problems,
    )

    keyword_results = retrieve_keyword_advice(
        advice_meta=advice_meta,
        keywords=keywords,
        top_k=top_keyword,
    )

    fused_results = fuse_results(
        advice_meta=advice_meta,
        graph_neo4j_results=graph_neo4j_results,
        graph_corpus_results=graph_corpus_results,
        vector_results=vector_results,
        keyword_results=keyword_results,
    )

    final_context = group_final_context(
        fused_results=fused_results,
        selected_problems=selected_problems,
        max_problem_advice=max_problem_advice,
        max_activity_advice=max_activity_advice,
        max_environment_advice=max_environment_advice,
    )

    elapsed = round(time.time() - start, 3)

    result: Dict[str, Any] = {
        "success": True,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": "v02_final_workflow_step05",
        "question": question,
        "selected_problem_ids": selected_problem_ids,
        "selected_problems": selected_problems,
        "mapping_result": mapping_result,
        "retrieval_counts": {
            "selected_problem_count": len(selected_problem_ids),
            "graph_neo4j_count": len(graph_neo4j_results),
            "graph_corpus_count": len(graph_corpus_results),
            "vector_advice_count": len(vector_results),
            "vector_skipped": vector_skipped,
            "graph_total_before_vector": graph_total_before_vector,
            "keyword_advice_count": len(keyword_results),
            "fused_count": len(fused_results),
            "final_problem_advice_count": len(final_context["problem_advice"]),
            "final_activity_advice_count": len(final_context["activity_advice"]),
            "final_environment_advice_count": len(final_context["environment_advice"]),
            "evidence_source_count": len(final_context["evidence_sources"]),
        },
        "retrieval_config": {
            "embedding_model": env["embedding_model"],
            "auto_map": auto_map,
            "top_advice_vector": top_advice_vector,
            "vector_skip_threshold_graph_results": 8,
            "top_keyword": top_keyword,
            "max_problem_advice": max_problem_advice,
            "max_activity_advice": max_activity_advice,
            "max_environment_advice": max_environment_advice,
        },
        "keywords": keywords,
        "final_context": final_context,
        "elapsed_seconds": elapsed,
    }

    if include_debug:
        result["debug"] = {
            "graph_neo4j_results": graph_neo4j_results[:100],
            "graph_corpus_results": [compact_advice_item(x) for x in graph_corpus_results[:100]],
            "vector_results": [compact_advice_item(x) for x in vector_results[:80]],
            "keyword_results": [compact_advice_item(x) for x in keyword_results[:80]],
            "fused_results": [compact_advice_item(x) for x in fused_results[:150]],
        }

    return result


def write_retrieval_result(result: Dict[str, Any]) -> Tuple[Path, Path]:
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_path = RETRIEVAL_DEBUG_DIR / f"retrieval_result_{timestamp}.json"

    content = json.dumps(result, ensure_ascii=False, indent=2)

    LAST_RETRIEVAL_PATH.write_text(content, encoding="utf-8")
    timestamp_path.write_text(content, encoding="utf-8")

    return LAST_RETRIEVAL_PATH, timestamp_path


def print_retrieval_preview(result: Dict[str, Any]) -> None:
    print_header("Retrieval preview")

    print(f"Question: {result.get('question')}")
    print(f"Selected problem IDs: {result.get('selected_problem_ids')}")
    print(f"Elapsed seconds: {result.get('elapsed_seconds')}")

    print("\nSelected problems:")
    for p in result.get("selected_problems") or []:
        confidence = p.get("confidence")
        confidence_text = f" confidence={confidence}" if confidence is not None else ""
        print(f"- {p.get('id')} [{p.get('category')}]: {p.get('content') or p.get('name')}{confidence_text}")

    print("\nRetrieval counts:")
    print(json.dumps(result.get("retrieval_counts"), ensure_ascii=False, indent=2))

    context = result.get("final_context") or {}

    def show_items(title: str, key: str, limit: int = 5) -> None:
        print("\n" + title)
        items = context.get(key) or []

        if not items:
            print("- None")
            return

        for item in items[:limit]:
            print("-" * 60)
            print(f"doc_id: {item.get('doc_id')}")
            print(f"score: {item.get('final_score')} | sources: {item.get('retrieval_sources')}")
            print(f"problem: {item.get('related_problem_ids')} / {item.get('related_problem_names')}")
            print(f"advice: {clean_text(item.get('advice_text'))[:220]}")
            print(f"source_title: {item.get('source_title')}")
            print(f"heading_path: {item.get('heading_path')}")

    show_items("ProblemAdvice top results:", "problem_advice")
    show_items("ActivityAdvice top results:", "activity_advice")
    show_items("EnvironmentAdvice top results:", "environment_advice")

    print("\nEvidence sources:")
    sources = context.get("evidence_sources") or []
    if not sources:
        print("- None")
    else:
        for src in sources[:5]:
            print("-" * 60)
            print(f"source_title: {src.get('source_title')}")
            print(f"heading_path: {src.get('heading_path')}")
            print(f"evidence_text: {clean_text(src.get('evidence_text'))[:220]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 05: Graph retrieval + FAISS vector retrieval + fusion."
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
        help="Optional mapped problem IDs, e.g. --problem_ids F3 H5. If omitted, mapper will be called.",
    )

    parser.add_argument(
        "--no_auto_map",
        action="store_true",
        help="Disable auto problem mapping when problem_ids are not provided.",
    )

    parser.add_argument("--top_advice_vector", type=int, default=DEFAULT_TOP_ADVICE_VECTOR)
    parser.add_argument("--top_keyword", type=int, default=DEFAULT_TOP_KEYWORD)
    parser.add_argument("--max_problem_advice", type=int, default=DEFAULT_MAX_PROBLEM_ADVICE)
    parser.add_argument("--max_activity_advice", type=int, default=DEFAULT_MAX_ACTIVITY_ADVICE)
    parser.add_argument("--max_environment_advice", type=int, default=DEFAULT_MAX_ENVIRONMENT_ADVICE)

    parser.add_argument(
        "--no_debug",
        action="store_true",
        help="Do not include debug details in saved result.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header("Step 05: Graph retrieval + FAISS vector retrieval + fusion")
    print("[READ-ONLY] This script reads Neo4j and local indexes. It does not modify KG.")
    print(f"Question: {args.question}")

    if args.problem_ids:
        print(f"Problem IDs: {args.problem_ids}")
    elif not args.no_auto_map:
        print("Problem IDs not provided. Auto-mapping with 04_problem_mapper.py will be used.")
    else:
        print("Problem IDs not provided and auto-mapping disabled.")

    result = retrieve(
        question=args.question,
        problem_ids=args.problem_ids,
        auto_map=not args.no_auto_map,
        top_advice_vector=args.top_advice_vector,
        top_keyword=args.top_keyword,
        max_problem_advice=args.max_problem_advice,
        max_activity_advice=args.max_activity_advice,
        max_environment_advice=args.max_environment_advice,
        include_debug=not args.no_debug,
    )

    last_path, timestamp_path = write_retrieval_result(result)

    print_retrieval_preview(result)

    print_header("Final result")
    print("[PASS] Step 05 retrieval finished.")
    print(f"[OK] Last retrieval result: {last_path}")
    print(f"[OK] Timestamp retrieval result: {timestamp_path}")
    print("Next step: write src/06_generate_answer.py to generate evidence-grounded care answer.")


if __name__ == "__main__":
    main()
