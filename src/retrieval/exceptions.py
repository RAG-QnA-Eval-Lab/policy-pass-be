"""Retrieval 계층 전용 예외 정의.

이 예외들은 route 핸들러에서 HTTPException으로 매핑됩니다.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Retrieval 모듈의 최상위 예외."""

    pass


class IndexLoadError(RetrievalError):
    """FAISS 인덱스 또는 metadata 로드/검증 실패 시 발생.

    Examples:
        - 파일 없음
        - 차원 불일치 (expected 1536 != actual)
        - ntotal 불일치 (index vs metadata)
        - 필수 메타데이터 키 누락
    """

    pass


class EmbeddingError(RetrievalError):
    """OpenAI 임베딩 API 호출 실패 시 발생.

    원인:
        - AuthenticationError (잘못된/누락 API 키)
        - RateLimitError
        - APIError / APIConnectionError
        - Timeout
    """

    pass


class SearchError(RetrievalError):
    """검색 실행 중 (FAISS search 또는 후처리) 실패 시 발생."""

    pass


__all__ = ["RetrievalError", "IndexLoadError", "EmbeddingError", "SearchError"]
