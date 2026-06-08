"""Policy Pass — FastAPI 백엔드 엔트리포인트.

Vector Search + RAG Answer Generation.
Lifespan에서 index + embedder + retriever + generator를 1회 초기화하여 app.state에 저장.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings

# API routes
from src.api.routes.rag import router as rag_router

# Generation core (RAG answer)
from src.generation.generator import OpenAIGenerator

# Retrieval core
from src.retrieval.embedder import OpenAIEmbedder
from src.retrieval.index_downloader import ensure_index_files
from src.retrieval.index_loader import load_index
from src.retrieval.retriever import VectorSearchRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: (optional S3 download) + FAISS 인덱스 + metadata + OpenAI embedder + Retriever + Generator 로드.

    ensure_index_files() is a no-op for local dev (DOWNLOAD_INDEX_FROM_S3=false).
    실패 시 앱 시작 중단 (fail-fast). production에서 필수.
    """
    logger.info("Policy Pass API starting (env=%s)", settings.environment)

    try:
        # 1. (Optional) S3 index download for production. No-op when DOWNLOAD_INDEX_FROM_S3=false.
        #    Must run before load_index so that local data/index or downloaded path is used.
        ensure_index_files()

        # 2. Vector index + metadata (pkl 우선, json fallback, 검증 포함)
        index, metadata = load_index()  # sync지만 startup에서 허용

        # 3. Embedder (OpenAI)
        embedder = OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

        # 4. Retriever (검색 로직 캡슐화)
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

        # 5. Generator (RAG answer generation via LangChain ChatOpenAI)
        generator = OpenAIGenerator(
            api_key=settings.openai_api_key,
            model=settings.chat_model,
            temperature=settings.temperature,
        )
        app.state.generator = generator
        logger.info("RAG generator ready (model=%s)", generator.model)

    except Exception as exc:
        logger.exception("CRITICAL: Vector index or RAG generator initialization failed")
        # 앱 시작 실패 (uWSGI / uvicorn이 비정상 종료 처리)
        raise RuntimeError("Failed to init vector search/RAG generator. Check logs + data/index/.") from exc

    yield

    # Shutdown (proper cleanup)
    retriever = getattr(app.state, "retriever", None)
    if retriever and hasattr(retriever.embedder, "close"):
        try:
            await retriever.embedder.close()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close OpenAI embedder", exc_info=True)

    generator = getattr(app.state, "generator", None)
    if generator and hasattr(generator, "close"):
        try:
            await generator.close()
        except Exception:  # noqa: BLE001
            logger.warning("Failed to close OpenAI generator", exc_info=True)

    logger.info("Policy Pass API shutting down")


app = FastAPI(
    title="Policy Pass API",
    description="청년정책 RAG QA 백엔드 (Vector Search + RAG Answer Generation)",
    version="0.3.0-rag-answer",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# RAG routes 등록 ( /api/v1/search + /api/v1/ask )
app.include_router(rag_router)


@app.get("/health")
async def health():
    """기본 헬스체크 + vector index 상태 (lifespan 로드 여부). RAG generator는 startup fail-fast로 보장."""
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
        "version": "0.3.0-rag-answer",
        "endpoints": ["/health", "/api/v1/search", "/api/v1/ask"],
    }
