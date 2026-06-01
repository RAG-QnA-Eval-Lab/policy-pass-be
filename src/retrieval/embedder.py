"""OpenAI Embedding 클라이언트 래퍼 (text-embedding-3-small 전용).

- settings.embedding_model의 "openai/" prefix 자동 제거
- AsyncOpenAI 사용 (FastAPI 비동기 친화)
- production: timeout, max_retries, 명확한 예외 매핑
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, OpenAIError

from .exceptions import EmbeddingError

logger = logging.getLogger(__name__)


def _normalize_model_name(model: str) -> str:
    """OpenAI SDK가 기대하는 모델명으로 정규화.

    Examples:
        "openai/text-embedding-3-small" -> "text-embedding-3-small"
        "text-embedding-3-small" -> "text-embedding-3-small"
    """
    if not model:
        return "text-embedding-3-small"
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


class OpenAIEmbedder:
    """OpenAI 임베딩 생성기.

    Usage:
        embedder = OpenAIEmbedder(api_key=..., model=settings.embedding_model)
        vec = await embedder.embed("청년 취업 지원")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            logger.warning("OpenAIEmbedder initialized with empty api_key — calls will fail")

        self.model = _normalize_model_name(model)
        self.client = AsyncOpenAI(
            api_key=api_key or "dummy",  # 실제 호출 시 에러 발생
            timeout=timeout,
            max_retries=max_retries,
        )
        logger.debug("OpenAIEmbedder ready (model=%s)", self.model)

    async def embed(self, text: str) -> list[float]:
        """단일 텍스트를 1536차원 벡터로 임베딩.

        Raises:
            EmbeddingError: OpenAI API 오류 또는 dimension mismatch
        """
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        from config.settings import settings
        expected_dim = settings.embedding_dim

        try:
            resp = await self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
            )
            vec = resp.data[0].embedding
            if len(vec) != 1536:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {len(vec)}"
                )
            return vec
        except OpenAIError as exc:
            if "authentication" in str(exc).lower() or "api key" in str(exc).lower():
                msg = "OpenAI API key invalid or missing. Set OPENAI_API_KEY in .env"
            elif "rate" in str(exc).lower():
                msg = "OpenAI rate limit exceeded. Retry later."
            else:
                msg = f"OpenAI embedding failed: {exc}"
            logger.error("Embedding error: %s", msg)
            raise EmbeddingError(msg) from exc
        except Exception as exc:
            logger.exception("Unexpected embedding error")
            raise EmbeddingError(f"Unexpected error during embedding: {exc}") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 (미래 hybrid/BM25 대비).

        OpenAI는 한 번에 여러 input 지원.
        """
        if not texts:
            return []
        # OpenAI SDK는 list[str]을 input으로 받음
        try:
            resp = await self.client.embeddings.create(
                model=self.model,
                input=[t.strip() for t in texts if t and t.strip()],
            )
            return [d.embedding for d in resp.data]
        except OpenAIError as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc

    async def close(self) -> None:
        """리소스 정리 (필요 시 lifespan shutdown에서 호출)."""
        await self.client.close()


__all__ = ["OpenAIEmbedder", "EmbeddingError"]
