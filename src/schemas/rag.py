"""RAG / Vector Search API 스키마 (Pydantic v2).

POST /api/v1/search 요청/응답 모델.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Vector search 요청.

    query는 최소 1자, 최대 500자. top_k는 1~100으로 제한 (기본은 settings.top_k).
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="검색 쿼리 텍스트 (예: '청년 취업 지원 정책')",
        examples=["청년 주거 지원", "광주 청년 구직활동수당"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="반환할 최대 결과 수 (기본값: settings.top_k, 최대 20)",
        examples=[5, 10],
    )

    model_config = {"json_schema_extra": {"example": {"query": "청년 취업 지원", "top_k": 5}}}


class SearchResultItem(BaseModel):
    """단일 검색 결과 아이템.

    distance: FAISS L2 distance (낮을수록 쿼리와 유사).
    content: 검색에 사용된 청크 텍스트 (전체 정책 요약/내용).
    """

    policy_id: str = Field(..., description="정책 고유 식별자")
    title: str = Field(..., description="정책 제목")
    content: str = Field(..., description="청크 본문 (검색 대상 텍스트)")
    category: str | None = Field(None, description="정책 카테고리 (employment, housing 등)")
    url: str | None = Field(None, description="원문 URL")
    last_updated: str | None = Field(None, description="마지막 업데이트 (YYYYMMDD)")
    chunk_index: int | None = Field(None, description="문서 내 청크 인덱스")
    source: str | None = Field(None, description="데이터 출처")
    distance: float = Field(
        ...,
        description="FAISS L2 distance (낮을수록 유사). IndexFlatL2 기준.",
        examples=[0.4123],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "policy_id": "20260420005400212772",
                "title": "(4차) 2026 광주청년 구직활동수당 및 활동지원사업",
                "content": "정책명: ... 요약: 광주 미취업 청년들의 ...",
                "category": "employment",
                "url": "",
                "last_updated": "20260601",
                "chunk_index": 0,
                "source": "data_portal",
                "distance": 0.4123,
            }
        }
    }


class SearchResponse(BaseModel):
    """Vector search 응답."""

    query: str = Field(..., description="원본 검색 쿼리")
    results: list[SearchResultItem] = Field(..., description="거리 오름차순 정렬된 결과")
    total_results: int = Field(..., description="반환된 결과 수")
    latency_ms: float | None = Field(
        None,
        description="서버 측 처리 소요 시간 (임베딩 + FAISS 검색, ms)",
        examples=[187.4],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "청년 취업 지원",
                "results": [
                    {
                        "policy_id": "...",
                        "title": "...",
                        "content": "...",
                        "category": "employment",
                        "distance": 0.4123,
                    }
                ],
                "total_results": 5,
                "latency_ms": 187.4,
            }
        }
    }


__all__ = ["SearchRequest", "SearchResultItem", "SearchResponse"]
