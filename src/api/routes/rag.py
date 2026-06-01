"""Vector Search 전용 API 라우트 (v1).

POST /api/v1/search — FAISS + OpenAI embedding 기반 semantic search.
(LLM 생성, hybrid, rerank 등은 이후 phase)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.retrieval.exceptions import EmbeddingError, SearchError
from src.schemas.rag import SearchRequest, SearchResponse, SearchResultItem

if TYPE_CHECKING:
    from src.retrieval import SearchResult
    from src.retrieval.retriever import VectorSearchRetriever

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


__all__ = ["router", "get_retriever"]
