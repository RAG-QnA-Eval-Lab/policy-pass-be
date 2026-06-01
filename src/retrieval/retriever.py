"""VectorSearchRetriever — FAISS + OpenAI 임베딩 기반 검색 로직.

production-ready:
- async search (embedding IO)
- top_k clamp
- distance 오름차순 보장 (FAISS L2)
- 상세 로깅 + timing
- SearchResult dataclass 반환
"""

from __future__ import annotations

import logging
import time
from typing import Any

import faiss
import numpy as np

from . import SearchResult
from .embedder import OpenAIEmbedder
from .exceptions import SearchError

logger = logging.getLogger(__name__)


class VectorSearchRetriever:
    """FAISS IndexFlatL2 기반 Vector Retriever (read-only).

    thread-safe for concurrent search (FAISS read-only 보장).
    """

    def __init__(
        self,
        index: faiss.Index,
        metadata: list[dict[str, Any]],
        embedder: OpenAIEmbedder,
        default_top_k: int = 10,
    ) -> None:
        if index.ntotal != len(metadata):
            raise ValueError("index.ntotal != len(metadata) at init")

        self.index = index
        self.metadata = metadata
        self.embedder = embedder
        self.default_top_k = default_top_k
        self._ntotal = index.ntotal
        self._dim = index.d

        logger.info(
            "VectorSearchRetriever initialized: %d vectors, dim=%d, default_top_k=%d",
            self._ntotal,
            self._dim,
            default_top_k,
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """쿼리를 임베딩하여 FAISS top-k 검색 수행.

        Args:
            query: 사용자 검색어 (빈 문자열/공백 금지)
            top_k: 반환할 최대 결과 수 (None이면 default, 1~ntotal로 clamp)

        Returns:
            SearchResult 리스트 (distance 오름차순 = 가장 유사한 순)

        Raises:
            SearchError: embedding 실패 또는 FAISS 오류
        """
        if not query or not query.strip():
            raise SearchError("Query must be non-empty string")

        k = top_k if top_k is not None else self.default_top_k
        k = max(1, min(k, self._ntotal))  # clamp

        t0 = time.perf_counter()

        try:
            # 1. 임베딩 (async)
            vec = await self.embedder.embed(query)

            # 2. FAISS search (CPU, 빠름)
            query_vec = np.asarray([vec], dtype=np.float32)
            distances, indices = self.index.search(query_vec, k)

            # 3. 결과 매핑
            results: list[SearchResult] = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # FAISS padding
                    continue

                idx_int = int(idx)
                if idx_int < 0 or idx_int >= len(self.metadata):
                    logger.warning("Invalid FAISS index returned: idx=%s", idx_int)
                    continue
                
                meta = self.metadata[int(idx)]
                result = SearchResult(
                    policy_id=meta.get("policy_id", ""),
                    title=meta.get("title", ""),
                    content=meta.get("content", ""),
                    category=meta.get("category"),
                    url=meta.get("url"),
                    last_updated=meta.get("last_updated"),
                    chunk_index=meta.get("chunk_index"),
                    source=meta.get("source"),
                    distance=float(dist),
                    metadata=meta,  # full 원본 (필요시)
                )
                results.append(result)

            # FAISS L2는 이미 오름차순이지만, 안전하게 재정렬
            results.sort(key=lambda r: r.distance)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "search q=%r k=%d -> %d results (%.1f ms, first_dist=%.4f)",
                query[:80],
                k,
                len(results),
                elapsed,
                results[0].distance if results else -1,
            )
            return results

        except Exception as exc:
            logger.exception("Search failed for query=%r", query[:50])
            if isinstance(exc, SearchError):
                raise
            raise SearchError(f"Vector search failed: {exc}") from exc

    def get_stats(self) -> dict[str, Any]:
        """헬스체크 / 모니터링용 통계."""
        return {
            "ntotal": self._ntotal,
            "dim": self._dim,
            "default_top_k": self.default_top_k,
            "model": getattr(self.embedder, "model", "unknown"),
        }


__all__ = ["VectorSearchRetriever"]
