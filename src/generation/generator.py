"""LangChain + ChatOpenAI 기반 RAG 답변 생성기 (OpenAIGenerator).

- Uses build_rag_answer_chain from src/api/chain.py (minimal LangChain layer)
- Keeps existing public interface: __init__ sig, generate(q, ctx)->(str,str), close()
- Model name normalization preserved (openai/gpt-4o-mini -> gpt-4o-mini)
- GenerationError wrapping preserved for API error mapping
- Context formatting (numbered block) done here before chain invoke
- No direct AsyncOpenAI chat calls; chain handles prompt|llm|parse
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from openai import OpenAIError

from src.api.chain import build_rag_answer_chain

from .exceptions import GenerationError

logger = logging.getLogger(__name__)


def _normalize_model_name(model: str) -> str:
    """OpenAI SDK가 기대하는 모델명으로 정규화.

    Examples:
        "openai/gpt-4o-mini" -> "gpt-4o-mini"
        "gpt-4o-mini" -> "gpt-4o-mini"
    """
    if not model:
        return "gpt-4o-mini"
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


class OpenAIGenerator:
    """LangChain ChatOpenAI wrapper for grounded RAG answer generation.

    Usage (unchanged):
        generator = OpenAIGenerator(api_key=..., model=settings.chat_model, temperature=settings.temperature)
        answer, model = await generator.generate(question, contexts)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            logger.warning("OpenAIGenerator initialized with empty api_key — calls will fail")

        self.model = _normalize_model_name(model)
        self.temperature = temperature

        # LangChain ChatOpenAI (uses async under the hood for ainvoke/ainvoke)
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=temperature,
            openai_api_key=api_key or "dummy",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.chain = build_rag_answer_chain(self.llm)

        logger.debug("OpenAIGenerator ready (model=%s, temperature=%s) [langchain]", self.model, temperature)

    async def generate(self, question: str, contexts: list[str]) -> tuple[str, str]:
        """질문 + 컨텍스트(이미 max_context_chars로 제한됨)를 바탕으로 grounded 답변 생성.

        Formats contexts as numbered block then invokes the LangChain chain.

        Returns:
            (answer: str, model: str)

        Raises:
            GenerationError: LLM/chain failure or invalid input
        """
        if not question or not question.strip():
            raise GenerationError("Question must be non-empty string")

        # contexts는 서비스 계층에서 이미 char 제한 + prefix 선택됨
        if not contexts:
            # short-circuit은 보통 서비스에서 하지만, 여기서도 안전장치
            answer = "제공된 컨텍스트에서 해당 정보를 찾을 수 없습니다."
            return answer, self.model

        # 컨텍스트 블록 구성 (번호 매김으로 LLM이 참조하기 쉽게) — caller per spec
        context_block = "\n\n".join(
            f"[{i+1}] {c.strip()}" for i, c in enumerate(contexts) if c and c.strip()
        )
        if not context_block:
            answer = "제공된 컨텍스트에서 해당 정보를 찾을 수 없습니다."
            return answer, self.model

        try:
            content = await self.chain.ainvoke(
                {"question": question.strip(), "contexts": context_block}
            )
            return (content or "").strip(), self.model

        except OpenAIError as exc:
            if "authentication" in str(exc).lower() or "api key" in str(exc).lower():
                msg = "OpenAI API key invalid or missing. Set OPENAI_API_KEY in .env"
            elif "rate" in str(exc).lower():
                msg = "OpenAI rate limit exceeded. Retry later."
            else:
                msg = f"OpenAI chat completion failed: {exc}"
            logger.error("Generation error: %s", msg)
            raise GenerationError(msg) from exc
        except Exception as exc:
            logger.exception("Unexpected generation error")
            raise GenerationError(f"Unexpected error during generation: {exc}") from exc

    async def close(self) -> None:
        """리소스 정리 (lifespan shutdown에서 호출). LangChain path: no-op (compat)."""
        # ChatOpenAI manages its async client internally; explicit close not required
        # for current usage, but method kept for interface stability with lifespan.
        pass


__all__ = ["OpenAIGenerator", "GenerationError"]
