"""RAG API routes (v1).

POST /api/v1/search — Vector Search (FAISS + OpenAI embedding)
POST /api/v1/ask — RAG answer generation (retriever.search 완전 재사용 + OpenAI Chat Completions)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.generation.exceptions import GenerationError
from src.retrieval.exceptions import EmbeddingError, SearchError
from src.schemas.rag import AskRequest, AskResponse, SearchRequest, SearchResponse, SearchResultItem, SourceItem

if TYPE_CHECKING:
    from src.generation.generator import OpenAIGenerator
    from src.retrieval import SearchResult
    from src.retrieval.retriever import VectorSearchRetriever
    from src.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


def get_retriever(request: Request) -> "VectorSearchRetriever":
    """lifespan에서 주입된 retriever를 반환하는 의존성.

    503 Service Unavailable: 인덱스 미로드 또는 로드 실패 시.
    """
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        logger.error("Vector retriever not available in app.state (startup failed?)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search service is temporarily unavailable. Index not loaded.",
        )
    return retriever


def get_generator(request: Request) -> "OpenAIGenerator":
    """lifespan에서 주입된 generator를 반환하는 의존성.

    503 Service Unavailable: generator 미로드 시.
    """
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        logger.error("RAG generator not available in app.state (startup failed?)")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG answer generation service is temporarily unavailable. Generator not loaded.",
        )
    return generator


def get_rag_service(request: Request) -> "RAGService":
    """retriever + generator를 조합한 RAGService 의존성.

    routes의 로직을 얇게 유지하기 위해 서비스 레이어에 위임.
    """
    retriever = get_retriever(request)
    generator = get_generator(request)
    # 지연 import로 순환 방지 (서비스는 generator/retriever를 런타임에만 필요)
    from src.services.rag_service import RAGService

    return RAGService(retriever=retriever, generator=generator)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Vector Search (FAISS + OpenAI embedding)",
    description="로컬 FAISS 인덱스를 사용한 semantic vector search만 수행합니다. "
    "LLM 답변, hybrid, reranker는 미구현 (1차 phase).",
)
async def search(
    payload: SearchRequest,
    retriever: "VectorSearchRetriever" = Depends(get_retriever),
) -> SearchResponse:
    """Vector search 엔드포인트.

    - query embedding → FAISS top-k
    - distance (L2) 오름차순 반환
    - latency 측정 포함
    """
    t0 = time.perf_counter()

    try:
        internal_results: list[SearchResult] = await retriever.search(
            query=payload.query,
            top_k=payload.top_k,
        )

        # 내부 SearchResult → API SearchResultItem 변환
        api_results: list[SearchResultItem] = [SearchResultItem(**r.to_dict()) for r in internal_results]

        latency = (time.perf_counter() - t0) * 1000

        logger.info(
            "search endpoint: query=%r top_k=%s results=%d latency=%.1fms",
            payload.query[:60],
            payload.top_k,
            len(api_results),
            latency,
        )

        return SearchResponse(
            query=payload.query,
            results=api_results,
            total_results=len(api_results),
            latency_ms=round(latency, 2),
        )

    except (EmbeddingError, SearchError) as exc:
        # 의존성 서비스 (OpenAI/FAISS) 문제 → 502
        logger.warning("Search dependency error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Vector search backend error: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /search")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during vector search.",
        ) from exc


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="RAG Answer Generation",
    description="retriever.search 재사용 + OpenAI Chat로 grounded 답변 생성. max_context_chars로 컨텍스트 크기 제한.",
)
async def ask(
    payload: AskRequest,
    service: "RAGService" = Depends(get_rag_service),
) -> AskResponse:
    """RAG 답변 생성 엔드포인트.

    - retriever.search(question, top_k) 완전 재사용 (embedding + FAISS)
    - max_context_chars 적용하여 실제 LLM에 주입할 contexts 제한 (relevance prefix)
    - ranked sources (rank 1-based + source + distance 등) + contexts 반환
    - latency 측정 + 로깅 포함
    """
    t0 = time.perf_counter()

    try:
        rag_result = await service.ask(
            question=payload.question,
            top_k=payload.top_k,
        )

        latency = (time.perf_counter() - t0) * 1000

        logger.info(
            "ask endpoint: question=%r top_k=%s contexts=%d model=%s latency=%.1fms",
            payload.question[:60],
            payload.top_k,
            len(rag_result.get("contexts", [])),
            rag_result.get("model"),
            latency,
        )

        # 서비스 dict → Pydantic 모델 변환 (SourceItem)
        api_sources: list[SourceItem] = [SourceItem(**s) for s in rag_result["sources"]]

        return AskResponse(
            question=payload.question,
            answer=rag_result["answer"],
            sources=api_sources,
            contexts=rag_result["contexts"],
            model=rag_result["model"],
            latency_ms=round(latency, 2),
        )

    except (EmbeddingError, SearchError, GenerationError) as exc:
        # 의존성 서비스 (OpenAI/FAISS/LLM) 문제 → 502
        logger.warning("Ask dependency error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"RAG answer generation backend error: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in /ask")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during RAG answer generation.",
        ) from exc


__all__ = ["router", "get_retriever", "get_generator", "get_rag_service"]
