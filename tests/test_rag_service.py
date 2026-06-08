"""Lightweight tests for RAGService using fakes.

- No OpenAI calls
- No real FAISS / VectorSearchRetriever / index
- Uses plain Python objects that duck-type the required async methods
- Uses asyncio.run (stdlib) instead of pytest-asyncio
- Patches settings via monkeypatch for deterministic context trimming
"""

import asyncio

import pytest

from src.retrieval import SearchResult
from src.services.rag_service import RAGService


class FakeRetriever:
    """Minimal duck-typed retriever for RAGService tests."""

    def __init__(self, results: list[SearchResult]):
        self.results = results
        self.search_calls = []

    async def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        self.search_calls.append((query, top_k))
        if not query or not query.strip():
            from src.retrieval.exceptions import SearchError

            raise SearchError("Query must be non-empty string")
        k = top_k if top_k is not None else 5
        return self.results[:k]


class FakeGenerator:
    """Minimal duck-typed generator."""

    def __init__(self, answer: str = "fake generated answer", model: str = "gpt-test-mini"):
        self.model = model
        self._answer = answer
        self.generate_calls = []

    async def generate(self, question: str, contexts: list[str]) -> tuple[str, str]:
        self.generate_calls.append((question, contexts))
        return self._answer, self.model


def test_ask_happy_path():
    results = [
        SearchResult(policy_id="p1", title="정책1", content="내용1 " * 5, distance=0.1),
        SearchResult(policy_id="p2", title="정책2", content="내용2 " * 5, distance=0.2, category="housing"),
    ]
    retr = FakeRetriever(results)
    gen = FakeGenerator(answer="이것은 테스트 답변입니다.", model="gpt-4o-mini")

    svc = RAGService(retriever=retr, generator=gen)
    result = asyncio.run(svc.ask("광주 청년 정책은?"))

    assert result["answer"] == "이것은 테스트 답변입니다."
    assert result["model"] == "gpt-4o-mini"
    assert len(result["contexts"]) == 2
    assert len(result["sources"]) == 2

    # sources must be in relevance order with rank
    assert result["sources"][0]["rank"] == 1
    assert result["sources"][0]["policy_id"] == "p1"
    assert result["sources"][1]["rank"] == 2
    assert result["sources"][1]["category"] == "housing"

    # retriever was called with default k=5 when None passed
    assert retr.search_calls[0] == ("광주 청년 정책은?", 5)


def test_ask_with_explicit_top_k():
    results = [
        SearchResult(policy_id=f"p{i}", title=f"t{i}", content=f"c{i}", distance=0.0 + i)
        for i in range(5)
    ]
    retr = FakeRetriever(results)
    gen = FakeGenerator()

    svc = RAGService(retriever=retr, generator=gen)
    result = asyncio.run(svc.ask("q", top_k=2))

    assert len(result["sources"]) == 2
    assert retr.search_calls[0][1] == 2


def test_ask_context_trimming(monkeypatch):
    """Test the max_context_chars budget logic inside RAGService."""
    # Use a tiny budget so we can exercise truncation
    monkeypatch.setattr("config.settings.settings.max_context_chars", 30)

    long_chunk = "가나다라마바사" * 10  # > 30 chars
    short_chunk = "짧은내용"

    results = [
        SearchResult(policy_id="p1", title="t1", content=long_chunk, distance=0.1),
        SearchResult(policy_id="p2", title="t2", content=short_chunk, distance=0.2),
        SearchResult(policy_id="p3", title="t3", content="또다른내용", distance=0.3),
    ]
    retr = FakeRetriever(results)
    gen = FakeGenerator(answer="trimmed answer", model="test-model")

    svc = RAGService(retriever=retr, generator=gen)
    result = asyncio.run(svc.ask("질문"))

    # Only the first (truncated) + possibly second should fit
    assert len(result["contexts"]) >= 1
    assert len(result["contexts"]) <= 2
    assert sum(len(c) for c in result["contexts"]) <= 30 + 1  # small slack for logic
    assert len(result["sources"]) == len(result["contexts"])
    # first context should be the prefix of the long one
    assert result["contexts"][0] == long_chunk[:30]


def test_ask_empty_results_short_circuit():
    retr = FakeRetriever([])
    gen = FakeGenerator(answer="should not be used", model="short-circuit-model")

    svc = RAGService(retriever=retr, generator=gen)
    result = asyncio.run(svc.ask("아무것도 없음"))

    assert "제공된 컨텍스트에서 해당 정보를 찾을 수 없습니다." in result["answer"]
    assert result["contexts"] == []
    assert result["sources"] == []
    assert result["model"] == "short-circuit-model"
    # generate should never have been called
    assert len(gen.generate_calls) == 0


def test_ask_skips_empty_content_results():
    results = [
        SearchResult(policy_id="p1", title="t1", content="", distance=0.1),
        SearchResult(policy_id="p2", title="t2", content="   \n\t  ", distance=0.2),
        SearchResult(policy_id="p3", title="t3", content="실제 내용", distance=0.3),
    ]
    retr = FakeRetriever(results)
    gen = FakeGenerator(answer="only real content", model="m1")

    svc = RAGService(retriever=retr, generator=gen)
    result = asyncio.run(svc.ask("q"))

    assert len(result["contexts"]) == 1
    assert result["contexts"][0] == "실제 내용"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["policy_id"] == "p3"


def test_ask_propagates_search_error():
    class ErrorRetriever:
        async def search(self, query, top_k=None):
            from src.retrieval.exceptions import SearchError
            raise SearchError("simulated embedding failure")

    svc = RAGService(retriever=ErrorRetriever(), generator=FakeGenerator())

    with pytest.raises(Exception) as excinfo:  # SearchError is subclass of Exception
        asyncio.run(svc.ask("bad query"))

    assert "simulated" in str(excinfo.value) or "SearchError" in str(type(excinfo.value))


def test_ask_uses_generator_model_on_short_circuit(monkeypatch):
    monkeypatch.setattr("config.settings.settings.max_context_chars", 10)
    gen = FakeGenerator(model="special-model-xyz")

    # Force the short-circuit path (where getattr(generator, "model") is used)
    # by using an empty retriever.
    retr2 = FakeRetriever([])
    svc2 = RAGService(retriever=retr2, generator=gen)
    result2 = asyncio.run(svc2.ask("q2"))
    assert result2["model"] == "special-model-xyz"
