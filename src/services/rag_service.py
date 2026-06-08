"""RAG orchestration service.

retrieval + generation을 조합하여 /ask 비즈니스 로직을 캡슐화.
routes는 HTTP glue + 의존성 + 에러 매핑 + 로깅만 담당하도록 함.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from src.generation.generator import OpenAIGenerator
from src.retrieval import SearchResult
from src.retrieval.retriever import VectorSearchRetriever

logger = logging.getLogger(__name__)


class RAGService:
    """Vector retrieval + LLM generation 오케스트레이션.

    - retriever.search(question, top_k) 100% 재사용 (요구사항)
    - max_context_chars를 사용해 LLM에 전달할 컨텍스트 char 수 제한 (prefix of relevance order)
    - ranked sources (rank 1-based + full metadata + distance) 구성
    - contexts 반환은 실제 LLM에 주입된 (trimmed) 값
    - empty context 시 LLM 호출 회피 + 한국어 fallback
    """

    def __init__(
        self,
        retriever: VectorSearchRetriever,
        generator: OpenAIGenerator,
    ) -> None:
        self.retriever = retriever
        self.generator = generator

    async def ask(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        """RAG 답변 생성 메인 진입점.

        Args:
            question: 사용자 질문
            top_k: 검색 결과 수 (None이면 5)

        Returns:
            {
                "answer": str,
                "sources": list[dict]  # SourceItem 필드와 호환 (rank 포함)
                "contexts": list[str],
                "model": str,
            }
        """
        k = top_k if top_k is not None else 5

        # 1. 기존 retriever 완전 재사용 (embedding + FAISS + 결과 정렬)
        results: list[SearchResult] = await self.retriever.search(
            query=question,
            top_k=k,
        )

        # 2. max_context_chars 적용 (relevance 순 prefix, char 누적 제한)
        # Safety: the actual contexts list (and sent to generator) total len must <= max_chars.
        # - first context can be truncated if it exceeds
        # - if a chunk exceeds remaining budget, take truncated prefix of its content
        # - sources and contexts built from exactly the same used results (in relevance order)
        limited_results: list[SearchResult] = []
        context_chunks: list[str] = []
        used_chars = 0
        max_chars = settings.max_context_chars

        for r in results:
            original = r.content or ""
            if not original.strip():
                continue
            if used_chars >= max_chars:
                break
            remaining = max_chars - used_chars
            if len(original) > remaining:
                # include truncated prefix of this content
                chunk = original[:remaining]
                limited_results.append(r)
                context_chunks.append(chunk)
                used_chars += len(chunk)
                break  # budget now exhausted
            else:
                limited_results.append(r)
                context_chunks.append(original)
                used_chars += len(original)

        # 3. sources (rank + 모든 요구 필드) + contexts 구성 from the same used results
        sources: list[dict[str, Any]] = []

        for i, r in enumerate(limited_results):
            sources.append(
                {
                    "rank": i + 1,
                    "policy_id": r.policy_id,
                    "title": r.title,
                    "category": r.category,
                    "source": r.source,
                    "url": r.url,
                    "last_updated": r.last_updated,
                    "chunk_index": r.chunk_index,
                    "distance": r.distance,
                }
            )

        contexts: list[str] = context_chunks

        # 4. 생성 (또는 short-circuit)
        if not contexts:
            answer = "제공된 컨텍스트에서 해당 정보를 찾을 수 없습니다."
            model = getattr(self.generator, "model", "unknown")
        else:
            answer, model = await self.generator.generate(question, contexts)

        logger.info(
            "RAGService.ask: question=%r top_k=%s retrieved=%d used=%d chars=%d model=%s",
            question[:60],
            k,
            len(results),
            len(limited_results),
            used_chars,
            model,
        )

        return {
            "answer": answer,
            "sources": sources,
            "contexts": contexts,
            "model": model,
        }


__all__ = ["RAGService"]
