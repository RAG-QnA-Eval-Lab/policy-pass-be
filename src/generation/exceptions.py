"""Generation 계층 전용 예외 (OpenAI Chat Completions 기반 RAG 답변 생성).

이 예외는 route 핸들러에서 HTTP 502로 매핑됩니다.
retrieval exceptions와 분리하여 retrieval 계층을 순수하게 유지 (요구사항 #4).
"""

from __future__ import annotations


class GenerationError(Exception):
    """OpenAI Chat Completions을 사용한 답변 생성 실패 시 발생.

    원인:
        - AuthenticationError (잘못된/누락 API 키)
        - RateLimitError
        - APIError / APIConnectionError / Timeout
        - Content filter 등 기타 OpenAI 오류
    """

    pass


__all__ = ["GenerationError"]
