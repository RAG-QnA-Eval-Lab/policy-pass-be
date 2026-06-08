"""Retrieval 패키지 — Vector Search 전용 (1차 구현).

SearchResult dataclass는 FAISS 검색 결과를 표현하는 내부 데이터 구조입니다.
API 응답으로는 schemas.rag.SearchResultItem으로 변환됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SearchResult:
    """FAISS vector search 결과 (내부용).

    distance는 L2 거리이며, 낮을수록 쿼리와 유사합니다.
    (FAISS IndexFlatL2 기준)

    Attributes:
        policy_id: 정책 고유 ID
        title: 정책 제목
        content: 청크 본문 (검색 대상 텍스트)
        category: 정책 카테고리 (employment, housing, welfare, education, participation 등)
        url: 원문 URL (없을 수 있음)
        last_updated: 마지막 업데이트 일자 (YYYYMMDD 형식 문자열)
        chunk_index: 문서 내 청크 순번 (0부터)
        source: 데이터 출처 (data_portal 등)
        distance: FAISS L2 distance (낮을수록 유사)
        metadata: 원본 메타데이터 전체 (필요 시 사용, 기본 None)
    """

    policy_id: str
    title: str
    content: str
    category: str | None = None
    url: str | None = None
    last_updated: str | None = None
    chunk_index: int | None = None
    source: str | None = None
    distance: float = 0.0
    metadata: dict[str, Any] | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Pydantic 모델 변환 또는 JSON 직렬화용 dict 반환."""
        return {
            "policy_id": self.policy_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "url": self.url,
            "last_updated": self.last_updated,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "distance": self.distance,
        }


__all__ = ["SearchResult"]
