"""RAG / Vector Search API 스키마 (Pydantic v2).

POST /api/v1/search — Vector Search
POST /api/v1/ask — RAG answer generation (retriever 재사용 + OpenAI Chat)
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


class AskRequest(BaseModel):
    """RAG 답변 생성 요청.

    retriever.search(question, top_k) 재사용 후 LLM으로 컨텍스트 기반 답변 생성.
    top_k는 /search와 동일한 1~20 범위, 기본값 5.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="사용자 질문 텍스트 (예: '광주 청년 구직활동수당 신청 방법')",
        examples=["광주 청년 구직활동수당이란?", "청년 월세 지원 정책은?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="검색 결과 수 (1~20, /search와 동일 범위; 기본 5; LLM 컨텍스트 char 제한으로 추가 trim 가능)",
        examples=[5, 10],
    )

    model_config = {"json_schema_extra": {"example": {"question": "광주 청년 구직활동수당이란?", "top_k": 5}}}


class SourceItem(BaseModel):
    """답변 생성에 사용된 출처 (ranked citation).

    rank + 전체 메타 + distance 포함. 실제 content는 contexts에 별도 반환 (투명성).
    """

    rank: int = Field(..., description="검색 순위 (1 = 가장 관련성 높음, relevance order)")
    policy_id: str = Field(..., description="정책 고유 식별자")
    title: str = Field(..., description="정책 제목")
    category: str | None = Field(None, description="정책 카테고리 (employment, housing 등)")
    source: str | None = Field(None, description="데이터 출처")
    url: str | None = Field(None, description="원문 URL")
    last_updated: str | None = Field(None, description="마지막 업데이트 (YYYYMMDD)")
    chunk_index: int | None = Field(None, description="문서 내 청크 인덱스")
    distance: float = Field(
        ...,
        description="FAISS L2 distance (낮을수록 유사). IndexFlatL2 기준.",
        examples=[0.4123],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "rank": 1,
                "policy_id": "20260420005400212772",
                "title": "(4차) 2026 광주청년 구직활동수당 및 활동지원사업",
                "category": "employment",
                "source": "data_portal",
                "url": None,
                "last_updated": "20260601",
                "chunk_index": 0,
                "distance": 0.4123,
            }
        }
    }


class AskResponse(BaseModel):
    """RAG 답변 생성 응답.

    question, answer, sources (ranked), contexts (실제 LLM 주입된 값), model, latency.
    """

    question: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="LLM이 생성한 답변 (제공된 contexts에 grounded)")
    sources: list[SourceItem] = Field(..., description="검색된 출처 (relevance 순 rank, char trim 적용된 subset)")
    contexts: list[str] = Field(..., description="LLM에 주입된 content (max_context_chars trim 후)")
    model: str = Field(..., description="답변 생성에 사용된 LLM 모델 (예: gpt-4o-mini)")
    latency_ms: float | None = Field(
        None,
        description="서버 측 전체 처리 소요 시간 (검색 + 컨텍스트 trim + 생성, ms)",
        examples=[1420.7],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "광주 청년 구직활동수당이란?",
                "answer": "광주 청년 구직활동수당은 미취업 청년에게 ... 지원합니다. 자세한 신청은 ...",
                "sources": [
                    {
                        "rank": 1,
                        "policy_id": "...",
                        "title": "...",
                        "category": "employment",
                        "source": "data_portal",
                        "distance": 0.4123,
                    }
                ],
                "contexts": ["정책명: ... 요약: ..."],
                "model": "gpt-4o-mini",
                "latency_ms": 1420.7,
            }
        }
    }


__all__ = [
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "AskRequest",
    "SourceItem",
    "AskResponse",
]
