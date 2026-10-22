"""
FastAPI server for dementia_care_graphrag_workflow_v01.

Minimal API version:
1. GET  /health
2. POST /chat

This API wraps src/08_graphrag_agent.py.
It does NOT modify Neo4j.
It reads local KG / FAISS indexes and writes local outputs / sessions.

Recommended architecture:
Mini Program / frontend
    -> FastAPI /chat
        -> GraphRAG Agent
            -> Neo4j + FAISS + DeepSeek
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

_AGENT_MODULE_CACHE = None
_PDF_MODULE_CACHE = None
_ICS_MODULE_CACHE = None


# =========================
# Basic helpers
# =========================

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_session_id(session_id: Optional[str]) -> str:
    sid = clean_text(session_id)
    if not sid:
        sid = "api_" + str(uuid.uuid4())[:12]
    sid = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", sid)
    return sid[:80] or ("api_" + str(uuid.uuid4())[:12])


def relative_path(path_like: Any) -> str:
    if not path_like:
        return ""
    path = Path(path_like)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def read_json(path_like: Any) -> Dict[str, Any]:
    if not path_like:
        return {}

    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def required_path_status() -> Dict[str, bool]:
    required = {
        "src/08_graphrag_agent.py": PROJECT_ROOT / "src" / "08_graphrag_agent.py",
        "src/05_retrieve.py": PROJECT_ROOT / "src" / "05_retrieve.py",
        "src/06_generate_answer.py": PROJECT_ROOT / "src" / "06_generate_answer.py",
        "src/07_session_memory.py": PROJECT_ROOT / "src" / "07_session_memory.py",
        "data/corpus/problem_catalog_v02_final.json": PROJECT_ROOT / "data" / "corpus" / "problem_catalog_v02_final.json",
        "data/corpus/kg_advice_corpus.jsonl": PROJECT_ROOT / "data" / "corpus" / "kg_advice_corpus.jsonl",
        "outputs/indexes/advice_bgem3_flatip.faiss": PROJECT_ROOT / "outputs" / "indexes" / "advice_bgem3_flatip.faiss",
        "outputs/indexes/advice_bgem3_meta.jsonl": PROJECT_ROOT / "outputs" / "indexes" / "advice_bgem3_meta.jsonl",
    }
    return {name: path.exists() for name, path in required.items()}


def load_agent_module():
    """
    Load and cache src/08_graphrag_agent.py in this API process.
    This keeps Step 05/06/07 runtime caches effective.
    """
    global _AGENT_MODULE_CACHE

    if _AGENT_MODULE_CACHE is not None:
        return _AGENT_MODULE_CACHE

    module_path = SRC_DIR / "08_graphrag_agent.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find {module_path}")

    spec = importlib.util.spec_from_file_location("step08_graphrag_agent_api", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    _AGENT_MODULE_CACHE = module
    return _AGENT_MODULE_CACHE


def load_tool_module(module_filename: str, module_name: str):
    module_path = SRC_DIR / module_filename

    if not module_path.exists():
        raise FileNotFoundError(f"Cannot find {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pdf_module():
    global _PDF_MODULE_CACHE

    if _PDF_MODULE_CACHE is not None:
        return _PDF_MODULE_CACHE

    _PDF_MODULE_CACHE = load_tool_module("09_pdf_generator.py", "step09_pdf_generator_api")
    return _PDF_MODULE_CACHE


def load_ics_module():
    global _ICS_MODULE_CACHE

    if _ICS_MODULE_CACHE is not None:
        return _ICS_MODULE_CACHE

    _ICS_MODULE_CACHE = load_tool_module("10_calendar_generator.py", "step10_calendar_generator_api")
    return _ICS_MODULE_CACHE


def resolve_project_file(path_like: Any) -> Path:
    """
    Resolve a user-supplied local project path safely.
    Only files under PROJECT_ROOT are allowed.
    """
    raw = clean_text(path_like)
    if not raw:
        raise ValueError("path is empty.")

    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    resolved = path.resolve()
    project_resolved = PROJECT_ROOT.resolve()

    if project_resolved not in resolved.parents and resolved != project_resolved:
        raise ValueError("Path is outside project root.")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    return resolved


def get_last_path_from_session(session_id: str, key: str) -> str:
    sid = safe_session_id(session_id)
    session_path = PROJECT_ROOT / "outputs" / "sessions" / f"{sid}.json"

    if not session_path.exists():
        return ""

    data = read_json(session_path)
    return clean_text(data.get(key))


# =========================
# Optional API token
# =========================

def verify_api_token(request: Request) -> None:
    """
    Optional protection.

    If API_AUTH_TOKEN is set in .env / environment,
    client must send:
        X-API-Key: <API_AUTH_TOKEN>

    If API_AUTH_TOKEN is empty, this check is disabled.
    """
    expected = clean_text(os.getenv("API_AUTH_TOKEN"))

    if not expected:
        return

    provided = clean_text(request.headers.get("X-API-Key"))

    if provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API token.")


# =========================
# Pydantic schemas
# =========================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="照护者输入的问题")
    session_id: Optional[str] = Field(default=None, description="会话 ID；不传则自动生成")
    reset_session: bool = Field(default=False, description="是否重置该 session")
    memory_context_mode: Literal["auto", "always", "off"] = Field(default="auto")
    include_active_problem_ids: bool = Field(default=False)
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(default=3200, ge=300, le=6000)


class ChatResponse(BaseModel):
    success: bool
    created_at: str
    session_id: str
    question: str
    used_memory_context: bool
    selected_problem_ids: List[str]
    answer_md: str
    retrieval_counts: Dict[str, Any]
    evidence_sources: List[Dict[str, Any]]
    paths: Dict[str, str]
    error: str = ""



class PdfRequest(BaseModel):
    answer_result_path: Optional[str] = Field(
        default=None,
        description="回答结果 JSON 路径；不传则尝试根据 session_id 找最近一次回答",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID；当 answer_result_path 为空时使用",
    )


class PdfResponse(BaseModel):
    success: bool
    created_at: str
    timestamp_pdf_path: str = ""
    last_pdf_path: str = ""
    download_url: str = ""
    error: str = ""


class IcsRequest(BaseModel):
    retrieval_result_path: Optional[str] = Field(
        default=None,
        description="检索结果 JSON 路径；不传则尝试根据 session_id 找最近一次检索",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话 ID；当 retrieval_result_path 为空时使用",
    )
    days: int = Field(default=7, ge=1, le=30, description="重复提醒天数")


class IcsResponse(BaseModel):
    success: bool
    created_at: str
    timestamp_ics_path: str = ""
    last_ics_path: str = ""
    download_url: str = ""
    event_count: int = 0
    events: List[Dict[str, Any]] = []
    error: str = ""


# =========================
# FastAPI app
# =========================

app = FastAPI(
    title="Dementia Care GraphRAG API",
    description="失智老人居家照护 GraphRAG 智能体 API",
    version="0.1.0",
)

cors_raw = os.getenv("API_CORS_ORIGINS", "*")
allow_origins = [x.strip() for x in cors_raw.split(",") if x.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    path_status = required_path_status()
    ok = all(path_status.values())

    return {
        "success": ok,
        "service": "dementia_care_graphrag_api",
        "version": "0.1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "required_paths": path_status,
        "message": "ok" if ok else "Some required files are missing.",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, _: None = Depends(verify_api_token)) -> Dict[str, Any]:
    question = clean_text(req.question)

    if not question:
        raise HTTPException(status_code=400, detail="question is empty.")

    session_id = safe_session_id(req.session_id)

    try:
        agent_module = load_agent_module()

        agent_result = agent_module.run_agent(
            session_id=session_id,
            question=question,
            problem_ids=None,
            memory_context_mode=req.memory_context_mode,
            include_active_problem_ids=req.include_active_problem_ids,
            reset_session=req.reset_session,
            no_auto_map=False,
            no_debug=True,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            show_steps=False,
            make_pdf=False,
            make_ics=False,
        )

        paths = agent_result.get("paths") or {}
        retrieval_path = paths.get("retrieval_timestamp") or paths.get("retrieval_last")
        retrieval_data = read_json(retrieval_path)

        final_context = retrieval_data.get("final_context") or {}
        evidence_sources = final_context.get("evidence_sources") or []

        response = {
            "success": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "question": question,
            "used_memory_context": bool(agent_result.get("used_memory_context")),
            "selected_problem_ids": agent_result.get("selected_problem_ids") or [],
            "answer_md": agent_result.get("answer_md") or "",
            "retrieval_counts": agent_result.get("retrieval_counts") or {},
            "evidence_sources": evidence_sources[:10],
            "paths": {
                "session": relative_path(paths.get("session")),
                "retrieval": relative_path(retrieval_path),
                "answer_md": relative_path(paths.get("answer_timestamp_md") or paths.get("answer_last_md")),
                "answer_json": relative_path(paths.get("answer_timestamp_json") or paths.get("answer_last_json")),
            },
            "error": "",
        }

        return response

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": repr(e),
                "hint": "Please check Neo4j, FAISS indexes, .env and DeepSeek API key.",
            },
        )



@app.post("/pdf", response_model=PdfResponse)
def make_pdf(req: PdfRequest, _: None = Depends(verify_api_token)) -> Dict[str, Any]:
    """
    Generate a PDF care plan from an answer_result JSON.

    Priority:
    1. req.answer_result_path
    2. session.last_answer_result_path
    3. outputs/answers/last_answer_result.json
    """
    try:
        answer_path = clean_text(req.answer_result_path)

        if not answer_path and req.session_id:
            answer_path = get_last_path_from_session(req.session_id, "last_answer_result_path")

        if not answer_path:
            answer_path = "outputs/answers/last_answer_result.json"

        resolved_answer_path = resolve_project_file(answer_path)

        pdf_module = load_pdf_module()
        result = pdf_module.generate_pdf_from_answer_result(resolved_answer_path)

        timestamp_pdf_path = relative_path(result.get("timestamp_pdf_path"))
        last_pdf_path = relative_path(result.get("last_pdf_path"))

        filename = Path(timestamp_pdf_path or last_pdf_path).name
        download_url = f"/download/pdfs/{filename}" if filename else ""

        return {
            "success": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp_pdf_path": timestamp_pdf_path,
            "last_pdf_path": last_pdf_path,
            "download_url": download_url,
            "error": "",
        }

    except Exception as e:
        return {
            "success": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp_pdf_path": "",
            "last_pdf_path": "",
            "download_url": "",
            "error": repr(e),
        }


@app.post("/ics", response_model=IcsResponse)
def make_ics(req: IcsRequest, _: None = Depends(verify_api_token)) -> Dict[str, Any]:
    """
    Generate ICS calendar reminders from a retrieval_result JSON.

    Priority:
    1. req.retrieval_result_path
    2. session.last_retrieval_result_path
    3. outputs/retrieval_debug/last_retrieval_result.json
    """
    try:
        retrieval_path = clean_text(req.retrieval_result_path)

        if not retrieval_path and req.session_id:
            retrieval_path = get_last_path_from_session(req.session_id, "last_retrieval_result_path")

        if not retrieval_path:
            retrieval_path = "outputs/retrieval_debug/last_retrieval_result.json"

        resolved_retrieval_path = resolve_project_file(retrieval_path)

        ics_module = load_ics_module()
        result = ics_module.generate_calendar_from_retrieval_result(
            retrieval_result_path=resolved_retrieval_path,
            days=req.days,
        )

        timestamp_ics_path = relative_path(result.get("timestamp_ics_path"))
        last_ics_path = relative_path(result.get("last_ics_path"))

        filename = Path(timestamp_ics_path or last_ics_path).name
        download_url = f"/download/calendars/{filename}" if filename else ""

        return {
            "success": True,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp_ics_path": timestamp_ics_path,
            "last_ics_path": last_ics_path,
            "download_url": download_url,
            "event_count": int(result.get("event_count", 0) or 0),
            "events": result.get("events") or [],
            "error": "",
        }

    except Exception as e:
        # For ICS, no inferred reminders is a normal product case.
        return {
            "success": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "timestamp_ics_path": "",
            "last_ics_path": "",
            "download_url": "",
            "event_count": 0,
            "events": [],
            "error": repr(e),
        }


@app.get("/download/{kind}/{filename}")
def download_file(kind: str, filename: str, _: None = Depends(verify_api_token)):
    """
    Download generated files.

    Allowed:
    - /download/pdfs/<filename>.pdf
    - /download/calendars/<filename>.ics
    """
    allowed_dirs = {
        "pdfs": PROJECT_ROOT / "outputs" / "pdfs",
        "calendars": PROJECT_ROOT / "outputs" / "calendars",
    }

    if kind not in allowed_dirs:
        raise HTTPException(status_code=404, detail="Unsupported download kind.")

    safe_name = Path(filename).name
    file_path = allowed_dirs[kind] / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")

    if kind == "pdfs":
        media_type = "application/pdf"
    else:
        media_type = "text/calendar"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
