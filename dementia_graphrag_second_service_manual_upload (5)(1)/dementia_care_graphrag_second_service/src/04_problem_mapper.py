"""
Step 04: Map caregiver natural-language question to 58 standard Problem nodes.

Inputs:
1. data/corpus/problem_catalog_v02_final.json
2. outputs/indexes/problem_bgem3_flatip.faiss
3. outputs/indexes/problem_bgem3_meta.jsonl
4. DeepSeek API through src/03_llm_client.py

Outputs:
1. Mapping result JSON printed to terminal
2. outputs/retrieval_debug/last_problem_mapping.json
3. outputs/retrieval_debug/problem_mapping_<timestamp>.json

This script does NOT read or modify Neo4j.
It does NOT retrieve Advice.
It only maps the user's question to Problem IDs.
"""

from __future__ import annotations


import argparse
import importlib.util
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PROBLEM_CATALOG_PATH = PROJECT_ROOT / "data" / "corpus" / "problem_catalog_v02_final.json"

INDEX_DIR = PROJECT_ROOT / "outputs" / "indexes"
PROBLEM_FAISS_PATH = INDEX_DIR / "problem_bgem3_flatip.faiss"
PROBLEM_META_PATH = INDEX_DIR / "problem_bgem3_meta.jsonl"

RETRIEVAL_DEBUG_DIR = PROJECT_ROOT / "outputs" / "retrieval_debug"
LAST_MAPPING_PATH = RETRIEVAL_DEBUG_DIR / "last_problem_mapping.json"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_VECTOR_TOP_K = 8


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


def load_deepseek_client_class():
    """
    Because the file name is 03_llm_client.py, it cannot be imported by normal syntax:
        from 03_llm_client import DeepSeekClient

    So we dynamically load it.
    """
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


def load_problem_catalog(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Problem catalog not found: {path}\n"
            "Please run: python src/01_prepare_kg_corpus.py"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("problem_catalog_v02_final.json should be a list.")

    normalized: List[Dict[str, Any]] = []

    for item in data:
        pid = clean_text(item.get("id")).upper()
        name = clean_text(item.get("name") or item.get("content"))
        content = clean_text(item.get("content") or item.get("name"))
        category = clean_text(item.get("category"))

        aliases = item.get("aliases") or item.get("alias") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        # Also try alias-like fields in properties.
        props = item.get("properties") or {}
        for key in ["alias", "aliases", "synonyms", "同义词", "别名"]:
            value = props.get(key)
            if not value:
                continue
            if isinstance(value, list):
                aliases.extend(value)
            else:
                aliases.append(value)

        aliases_clean = []
        seen_alias = set()
        for alias in aliases:
            alias_text = clean_text(alias)
            if alias_text and alias_text not in seen_alias:
                aliases_clean.append(alias_text)
                seen_alias.add(alias_text)

        normalized.append(
            {
                "id": pid,
                "name": name,
                "content": content,
                "category": category,
                "alias": aliases_clean,
                "properties": props,
            }
        )

    return normalized


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
    #return SentenceTransformer('/root/models--BAAI--bge-m3')


def query_instruction(query: str) -> str:
    query = clean_text(query)
    return f"为这个句子生成表示以用于检索相关照护问题：{query}"


def embed_query(model, query: str) -> np.ndarray:
    q_emb = model.encode(
        [query_instruction(query)],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    return q_emb


class LLMProblemMapper:
    """
    Map caregiver natural language question to standard Problem IDs.

    Key rules:
    1. DeepSeek can only choose IDs from local problem catalog.
    2. IDs are validated locally.
    3. content/category/name are overwritten by local catalog.
    4. FAISS Problem candidates are only used as hints.
    5. Final output is safe for downstream retrieval.
    """

    def __init__(
        self,
        catalog_path: Path = PROBLEM_CATALOG_PATH,
        enable_vector_candidates: bool = True,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
    ) -> None:
        load_dotenv(PROJECT_ROOT / ".env")

        self.catalog_path = catalog_path
        self.problem_catalog = load_problem_catalog(catalog_path)
        self.problem_lookup = {item["id"]: item for item in self.problem_catalog}

        self.enable_vector_candidates = enable_vector_candidates
        self.vector_top_k = vector_top_k

        DeepSeekClient = load_deepseek_client_class()
        self.llm = DeepSeekClient()

        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._embedding_model = None
        self._faiss_index = None
        self._problem_meta = None

    def _lazy_load_vector_assets(self) -> bool:
        """
        Load FAISS Problem index and BGE-M3 model only when needed.

        If vector assets are missing, mapper still works with full 58-catalog prompt.
        """
        if not self.enable_vector_candidates:
            return False

        if self._embedding_model is not None and self._faiss_index is not None and self._problem_meta is not None:
            return True

        if not PROBLEM_FAISS_PATH.exists() or not PROBLEM_META_PATH.exists():
            print("[WARN] Problem FAISS index or metadata not found. Vector candidates disabled.")
            return False

        try:
            faiss = import_faiss()
            self._faiss_index = faiss.read_index(str(PROBLEM_FAISS_PATH))
            self._problem_meta = load_jsonl(PROBLEM_META_PATH)
            self._embedding_model = load_embedding_model(self.embedding_model_name)
            return True
        except Exception as e:
            print(f"[WARN] Failed to load vector assets: {repr(e)}")
            return False

    def retrieve_vector_candidates(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve candidate Problem nodes using FAISS.
        """
        if not self._lazy_load_vector_assets():
            return []

        top_k = top_k or self.vector_top_k
        top_k = min(top_k, len(self._problem_meta))

        q_emb = embed_query(self._embedding_model, question)
        print("q_emb:",type( q_emb))

        scores, indices = self._faiss_index.search(q_emb, top_k)

        candidates: List[Dict[str, Any]] = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self._problem_meta):
                continue

            meta = self._problem_meta[int(idx)]
            pid = clean_text(meta.get("id")).upper()

            local_item = self.problem_lookup.get(pid, {})
            if not local_item:
                continue

            candidates.append(
                {
                    "id": pid,
                    "content": local_item.get("content") or local_item.get("name"),
                    "name": local_item.get("name") or local_item.get("content"),
                    "category": local_item.get("category"),
                    "vector_score": float(score),
                    "rank": rank,
                }
            )

        return candidates

    def build_catalog_text(self) -> str:
        """
        Build complete 58 Problem catalog text for LLM.
        """
        lines: List[str] = []

        current_category = None

        for item in self.problem_catalog:
            category = item.get("category", "")
            if category != current_category:
                current_category = category
                lines.append(f"\n【{category}】")

            alias_list = item.get("alias") or []
            alias_text = ""
            if alias_list:
                alias_text = " | 别名/同义表达: " + "；".join(alias_list)

            lines.append(
                f'- ID: {item["id"]} | 问题: {item["content"]}{alias_text}'
            )

        return "\n".join(lines).strip()

    def build_vector_candidate_text(self, candidates: List[Dict[str, Any]]) -> str:
        if not candidates:
            return "无。"

        lines: List[str] = []
        for item in candidates:
            score = item.get("vector_score", 0.0)
            try:
                score_text = f"{float(score):.4f}"
            except Exception:
                score_text = ""

            lines.append(
                f'- rank={item.get("rank")} | score={score_text} | '
                f'ID: {item.get("id")} | 类别: {item.get("category")} | 问题: {item.get("content")}'
            )

        return "\n".join(lines)

    def build_system_prompt(self) -> str:
        return """
你是一个“失智老人居家照护知识图谱问题映射器”。

你的任务：
根据照护者输入的自然语言问题，从候选问题目录中选择最匹配的一个或多个照护问题编码。

严格规则：
1. 只能从候选问题目录中选择 ID，绝对不能编造新 ID。
2. 如果一句话中包含多个照护问题，可以返回多个 ID。
3. 如果问题描述不够明确，matched_problems 返回空列表，并给出需要追问的问题。
4. 只输出 JSON，不要输出解释性文字。
5. content、category 必须以候选目录为准，不要自行改写。
6. 不要把“原因推测”当作问题编码，必须根据用户描述的实际表现进行映射。
7. 如果用户描述的是同义说法，应优先参考问题原文和照护语境进行归一化。
8. 如果涉及紧急风险，如噎食、严重呛咳、误服、自伤、伤人、跌倒后明显异常，应优先映射到安全风险相关问题。
9. 如果用户同时描述睡眠、精神行为、进食、排泄、安全风险等多个问题，应全部返回。
10. 向量候选问题只作为参考，不等于最终答案；最终仍然必须从完整候选目录中选择。
11. 如果“夜间多次起床”和“昼夜颠倒”同时出现，要根据原文区分：夜间频繁醒来/起床偏 F2；白天睡、晚上精神偏 F3；两者都有则都选。
12. 如果“说有人偷东西、被害、怀疑家人害他”等固定错误信念，优先考虑 H14 妄想。
13. 如果“看见不存在的人或东西、听见不存在的声音”，优先考虑 H13 幻觉。
14. 如果“喝水呛、吃饭呛”，优先考虑 I9 呛咳；如果“卡住、噎住、窒息风险”，优先考虑 I10 噎食；两者都有则都选。
15. 如果“往外跑”是行为本身，优先考虑 I5 外跑；如果“找不到家、走丢”，优先考虑 I6 走失；两者都有则都选。
"""

    def build_user_prompt(
        self,
        question: str,
        vector_candidates: List[Dict[str, Any]],
    ) -> str:
        return f"""
以下是向量检索初步认为可能相关的候选问题，仅供参考：
{self.build_vector_candidate_text(vector_candidates)}

以下是完整候选问题目录，最终答案只能从这里选择：
{self.build_catalog_text()}

照护者输入的问题：
{question}

请严格输出以下 JSON 格式：
{{
  "matched_problems": [
    {{
      "id": "候选目录中的ID",
      "content": "候选目录中的问题原文",
      "category": "候选目录中的类别",
      "confidence": 0.0,
      "evidence_in_question": "用户原文中支持该映射的片段",
      "reason": "为什么映射到这个问题，简短说明"
    }}
  ],
  "need_clarification": false,
  "clarifying_question": "",
  "unmapped_description": ""
}}
"""

    @staticmethod
    def parse_json_safely(raw: str) -> Dict[str, Any]:
        """
        Parse JSON string safely.

        The client already strips markdown fences, but this adds one more fallback:
        extract the first JSON object if extra text appears.
        """
        raw = clean_text(raw)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("Cannot parse JSON from LLM output", raw, 0)

    def map_question(
        self,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 2200,
    ) -> Dict[str, Any]:
        """
        Input natural language question, output validated mapping result.
        """
        start = time.time()
        question = clean_text(question)

        if not question:
            return {
                "success": False,
                "matched_problems": [],
                "need_clarification": True,
                "clarifying_question": "请先输入需要咨询的照护问题。",
                "unmapped_description": "",
                "original_question": question,
                "elapsed_seconds": 0.0,
            }

        vector_candidates = self.retrieve_vector_candidates(question, top_k=self.vector_top_k)

        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(
            question=question,
            vector_candidates=vector_candidates,
        )

        try:
            raw = self.llm.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            parsed = self.parse_json_safely(raw)

        except Exception as e:
            return {
                "success": False,
                "matched_problems": [],
                "need_clarification": True,
                "clarifying_question": "我暂时没有成功识别出具体照护问题，请补充描述老人的具体表现、发生频率和发生场景。",
                "unmapped_description": question,
                "original_question": question,
                "vector_candidates": vector_candidates,
                "error": repr(e),
                "elapsed_seconds": round(time.time() - start, 3),
            }

        result = self.validate_mapping(
            parsed=parsed,
            question=question,
            vector_candidates=vector_candidates,
        )

        result["elapsed_seconds"] = round(time.time() - start, 3)
        return result

    def validate_mapping(
        self,
        parsed: Dict[str, Any],
        question: str,
        vector_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Local validation:
        - remove nonexistent IDs
        - deduplicate
        - overwrite content/category/name from local catalog
        - clamp confidence into 0~1
        """
        matched = parsed.get("matched_problems", [])

        if not isinstance(matched, list):
            matched = []

        valid_results: List[Dict[str, Any]] = []
        seen_ids = set()

        vector_candidate_ids = [item.get("id") for item in vector_candidates]

        for item in matched:
            if not isinstance(item, dict):
                continue

            pid = clean_text(item.get("id")).upper()

            if pid not in self.problem_lookup:
                continue

            if pid in seen_ids:
                continue

            catalog_item = self.problem_lookup[pid]

            confidence = item.get("confidence", 0.0)
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.0

            confidence = max(0.0, min(1.0, confidence))

            result_item = {
                "id": pid,
                "name": catalog_item["name"],
                "content": catalog_item["content"],
                "category": catalog_item["category"],
                "confidence": confidence,
                "evidence_in_question": clean_text(item.get("evidence_in_question")),
                "reason": clean_text(item.get("reason")),
                "in_vector_candidates": pid in vector_candidate_ids,
            }

            alias = catalog_item.get("alias") or []
            if alias:
                result_item["alias"] = alias

            valid_results.append(result_item)
            seen_ids.add(pid)

        need_clarification = bool(parsed.get("need_clarification", False))

        if not valid_results:
            need_clarification = True

        return {
            "success": True,
            "matched_problems": valid_results,
            "matched_problem_ids": [item["id"] for item in valid_results],
            "need_clarification": need_clarification,
            "clarifying_question": clean_text(
                parsed.get(
                    "clarifying_question",
                    "请补充描述老人的具体表现、发生时间、频率和诱因。"
                )
            ),
            "unmapped_description": clean_text(parsed.get("unmapped_description")),
            "original_question": question,
            "vector_candidates": vector_candidates,
            "catalog_path": str(self.catalog_path.relative_to(PROJECT_ROOT)),
        }


def write_mapping_result(result: Dict[str, Any]) -> Tuple[Path, Path]:
    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_path = RETRIEVAL_DEBUG_DIR / f"problem_mapping_{timestamp}.json"

    content = json.dumps(result, ensure_ascii=False, indent=2)

    LAST_MAPPING_PATH.write_text(content, encoding="utf-8")
    timestamp_path.write_text(content, encoding="utf-8")

    return LAST_MAPPING_PATH, timestamp_path


def print_mapping_preview(result: Dict[str, Any]) -> None:
    print_header("Problem mapping preview")

    print(f"Question: {result.get('original_question')}")
    print(f"Success: {result.get('success')}")
    print(f"Need clarification: {result.get('need_clarification')}")
    print(f"Elapsed seconds: {result.get('elapsed_seconds')}")

    print("\nVector candidates:")
    for item in result.get("vector_candidates", [])[:8]:
        score = item.get("vector_score")
        try:
            score_text = f"{float(score):.4f}"
        except Exception:
            score_text = ""
        print(
            f"- rank={item.get('rank')} score={score_text} "
            f"{item.get('id')} [{item.get('category')}]: {item.get('content')}"
        )

    print("\nMatched problems:")
    matched = result.get("matched_problems") or []
    if not matched:
        print("- None")
    else:
        for item in matched:
            print("-" * 60)
            print(f"ID: {item.get('id')}")
            print(f"Category: {item.get('category')}")
            print(f"Problem: {item.get('content')}")
            print(f"Confidence: {item.get('confidence')}")
            print(f"Evidence: {item.get('evidence_in_question')}")
            print(f"Reason: {item.get('reason')}")

    if result.get("need_clarification"):
        print("\nClarifying question:")
        print(result.get("clarifying_question"))


def run_examples() -> Dict[str, Any]:
    mapper = LLMProblemMapper()

    test_questions = [
        "奶奶晚上总起来，还说有人偷她的钱，怎么办？",
        "爷爷不愿意洗澡，一帮他洗澡就打人。",
        "老人喝水总是呛，有时候吃饭像卡住了一样。",
        "奶奶总往外跑，出去之后找不到家。",
        "老人白天睡很多，晚上特别精神，还一直走来走去。",
        "爷爷不愿意洗澡，一帮他洗澡就打人，而且白天睡很多，晚上特别精神，还一直走来走去，怎么办？",
    ]

    results: List[Dict[str, Any]] = []

    for question in test_questions:
        print_header(f"Example: {question}")
        result = mapper.map_question(question)
        print_mapping_preview(result)
        results.append(result)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_count": len(results),
        "results": results,
    }

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 04: Map natural language question to 58 standard Problem IDs."
    )

    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="A caregiver question to map.",
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run built-in example questions.",
    )

    parser.add_argument(
        "--no_vector",
        action="store_true",
        help="Disable FAISS Problem candidates.",
    )

    parser.add_argument(
        "--vector_top_k",
        type=int,
        default=DEFAULT_VECTOR_TOP_K,
        help="Top-k FAISS Problem candidates.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print_header("Step 04: Problem mapper")
    print("[READ-ONLY] This script maps user questions to Problem IDs. It does not modify KG.")

    if args.examples or not args.question:
        report = run_examples()
        last_path, timestamp_path = write_mapping_result(report)

        print_header("Final result")
        print("[PASS] Step 04 example mapping finished.")
        print(f"[OK] Last mapping report: {last_path}")
        print(f"[OK] Timestamp mapping report: {timestamp_path}")
        print("Next step: write retrieval script that accepts matched_problem_ids.")

    else:
        mapper = LLMProblemMapper(
            enable_vector_candidates=not args.no_vector,
            vector_top_k=args.vector_top_k,
        )

        result = mapper.map_question(args.question)
        last_path, timestamp_path = write_mapping_result(result)

        print_mapping_preview(result)

        print_header("Final result")
        print("[PASS] Step 04 problem mapping finished.")
        print(f"[OK] Last mapping: {last_path}")
        print(f"[OK] Timestamp mapping: {timestamp_path}")
        print("Next step: pass matched_problem_ids into retrieval.")


if __name__ == "__main__":
    main()
