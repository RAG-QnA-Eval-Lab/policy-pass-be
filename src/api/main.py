"""Policy Pass — FastAPI 백엔드 엔트리포인트.

Vector Search Only (1차): FAISS + OpenAI embedding.
Lifespan에서 index_loader + retriever를 1회 초기화하여 app.state에 저장.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings

# API routes
from src.api.routes.rag import router as rag_router

# Retrieval core (vector search only)
from src.retrieval.embedder import OpenAIEmbedder
from src.retrieval.index_loader import load_index
from src.retrieval.retriever import VectorSearchRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: FAISS 인덱스 + metadata + OpenAI embedder + Retriever 로드.

    실패 시 앱 시작 중단 (fail-fast). production에서 필수.
    """
    logger.info("Policy Pass API starting (env=%s)", settings.environment)

    try:
        # 1. Vector index + metadata (pkl 우선, json fallback, 검증 포함)
        index, metadata = load_index()  # sync지만 startup에서 허용

        # 2. Embedder (OpenAI)
        embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

        # 3. Retriever (검색 로직 캡슐화)
        retriever = VectorSearchRetriever(
            index=index,
            metadata=metadata,
            embedder=embedder,
            default_top_k=settings.top_k,
        )

        app.state.retriever = retriever
        logger.info(
            "Vector search ready: %d vectors (model=%s)",
            len(metadata),
            embedder.model,
        )

    except Exception as exc:
        logger.exception("CRITICAL: Vector index initialization failed")
        # 앱 시작 실패 (uWSGI / uvicorn이 비정상 종료 처리)
        raise RuntimeError("Failed to initialize vector search. Check logs and data/index/ files.") from exc

    yield

    # Shutdown (proper cleanup)
    retriever = getattr(app.state, "retriever", None)
    if retriever and hasattr(retriever.embedder, "close"):
        try:
            await retriever.embedder.close()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close OpenAI embedder", exc_info=True)
    logger.info("Policy Pass API shutting down")


app = FastAPI(
    title="Policy Pass API",
    description="청년정책 RAG QA 백엔드 (Vector Search Only 1차 구현)",
    version="0.2.0-vector-search",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Vector Search 라우트 등록 ( /api/v1/search )
app.include_router(rag_router)


@app.get("/health")
async def health():
    """기본 헬스체크 + vector index 상태 (lifespan 로드 여부)."""
    retriever = getattr(app.state, "retriever", None)
    if retriever is None:
        return {"status": "degraded", "vector_index": "not_loaded"}
    stats = retriever.get_stats()
    return {
        "status": "ok",
        "vector_index": "loaded",
        "vectors": stats.get("ntotal"),
        "dim": stats.get("dim"),
    }


@app.get("/")
async def root():
    return {
        "message": "Policy Pass API",
        "version": "0.2.0-vector-search",
        "endpoints": ["/health", "/api/v1/search"],
    }
