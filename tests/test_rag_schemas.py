"""Tests for RAG schemas (Pydantic models).

These are pure, synchronous, no external dependencies or side effects.
"""

import pytest
from pydantic import ValidationError

from src.schemas.rag import (
    AskRequest,
    AskResponse,
    SearchRequest,
    SearchResultItem,
    SourceItem,
)


def test_ask_request_valid():
    req = AskRequest(question="광주 청년 구직활동수당이란?", top_k=5)
    assert req.question == "광주 청년 구직활동수당이란?"
    assert req.top_k == 5

    # default top_k
    req2 = AskRequest(question="test")
    assert req2.top_k == 5


def test_ask_request_validation_errors():
    # empty question
    with pytest.raises(ValidationError):
        AskRequest(question="")

    # too long
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 501)

    # top_k out of range
    with pytest.raises(ValidationError):
        AskRequest(question="q", top_k=0)

    with pytest.raises(ValidationError):
        AskRequest(question="q", top_k=21)


def test_search_request_validation():
    req = SearchRequest(query="청년 주거 지원", top_k=10)
    assert req.query == "청년 주거 지원"

    with pytest.raises(ValidationError):
        SearchRequest(query="")

    with pytest.raises(ValidationError):
        SearchRequest(query="q", top_k=25)


def test_source_item_valid():
    item = SourceItem(
        rank=1,
        policy_id="20260420005400212772",
        title="구직활동수당",
        distance=0.4123,
    )
    assert item.rank == 1
    assert item.category is None
    assert item.distance == 0.4123


def test_source_item_requires_rank_policy_id_title_distance():
    with pytest.raises(ValidationError):
        SourceItem(rank=1, policy_id="p1", title="t")  # missing distance


def test_ask_response_valid():
    source = SourceItem(rank=1, policy_id="p1", title="t1", distance=0.1)
    resp = AskResponse(
        question="q?",
        answer="답변입니다.",
        sources=[source],
        contexts=["정책 내용 chunk"],
        model="gpt-4o-mini",
        latency_ms=123.4,
    )
    assert resp.question == "q?"
    assert len(resp.sources) == 1
    assert resp.contexts == ["정책 내용 chunk"]
    assert resp.latency_ms == 123.4


def test_ask_response_from_service_dict_shape():
    """Verify that the dict shape produced by RAGService is usable to build AskResponse
    (mirrors what src/api/routes/rag.py does, without touching routes).
    """
    service_dict = {
        "answer": "테스트 답변",
        "sources": [
            {
                "rank": 1,
                "policy_id": "pid123",
                "title": "정책 제목",
                "category": "employment",
                "source": "data_portal",
                "url": None,
                "last_updated": "20260601",
                "chunk_index": 0,
                "distance": 0.123,
            }
        ],
        "contexts": ["chunk1", "chunk2"],
        "model": "gpt-4o-mini",
    }

    # Build sources the way the route does
    api_sources = [SourceItem(**s) for s in service_dict["sources"]]

    resp = AskResponse(
        question="원본 질문",
        answer=service_dict["answer"],
        sources=api_sources,
        contexts=service_dict["contexts"],
        model=service_dict["model"],
        latency_ms=None,
    )
    assert resp.answer == "테스트 답변"
    assert resp.sources[0].policy_id == "pid123"


def test_ask_response_empty_contexts_allowed():
    """Short-circuit case produces empty contexts list."""
    source = SourceItem(rank=1, policy_id="p1", title="t", distance=0.0)
    resp = AskResponse(
        question="q",
        answer="제공된 컨텍스트에서 해당 정보를 찾을 수 없습니다.",
        sources=[source],  # even in short-circuit the route may still provide sources? but contexts empty
        contexts=[],
        model="gpt-4o-mini",
    )
    assert resp.contexts == []


def test_search_result_item_valid():
    item = SearchResultItem(
        policy_id="p1",
        title="t",
        content="content here",
        distance=0.5,
    )
    assert item.distance == 0.5
    assert item.category is None
