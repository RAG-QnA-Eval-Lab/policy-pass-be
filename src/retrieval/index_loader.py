"""FAISS 인덱스 + metadata 로더 (lifespan에서 1회 호출).

- metadata.pkl 우선 로드, 실패 시 metadata.json fallback
- 강력한 startup validation (dim, ntotal, 필수 키)
- production: 명확한 에러 메시지 + logging + timing
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import faiss

from config.settings import settings

from .exceptions import IndexLoadError

logger = logging.getLogger(__name__)

# 메타데이터에서 반드시 존재해야 하는 필드 (최소 검증)
REQUIRED_META_KEYS: set[str] = {"policy_id", "content", "title"}


def _resolve_index_dir(index_dir: Path | str | None = None) -> Path:
    """index_dir를 Path로 정규화. None이면 settings.index_dir 사용."""
    if index_dir is None:
        index_dir = settings.index_dir
    return Path(index_dir).resolve()


def _load_metadata_pkl(pkl_path: Path) -> list[dict[str, Any]]:
    """pickle 파일 로드 (protocol 기본)."""
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise IndexLoadError(f"metadata.pkl must be list, got {type(data)}")
    return data


def _load_metadata_json(json_path: Path) -> list[dict[str, Any]]:
    """JSON 파일 로드."""
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise IndexLoadError(f"metadata.json must be list, got {type(data)}")
    return data


def _load_metadata(index_dir: Path) -> list[dict[str, Any]]:
    """pkl 우선, 실패 시 json fallback.

    Raises:
        IndexLoadError: 둘 다 실패하거나 데이터가 비정상일 때
    """
    pkl_path = index_dir / "metadata.pkl"
    json_path = index_dir / "metadata.json"

    # 1) pkl 시도
    if pkl_path.exists():
        try:
            meta = _load_metadata_pkl(pkl_path)
            logger.info("Loaded metadata from %s (%d items)", pkl_path.name, len(meta))
            return meta
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s (%s), falling back to metadata.json",
                pkl_path.name,
                exc,
            )

    # 2) json fallback
    if json_path.exists():
        try:
            meta = _load_metadata_json(json_path)
            logger.info("Loaded metadata from %s (%d items) [fallback]", json_path.name, len(meta))
            return meta
        except Exception as exc:
            raise IndexLoadError(f"Failed to load metadata.json: {exc}") from exc

    raise IndexLoadError(f"No metadata file found in {index_dir}. Expected metadata.pkl or metadata.json.")


def _validate_index_and_metadata(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
) -> None:
    """인덱스와 메타데이터 간 일치성 + 필수 구조 검증.

    Raises:
        IndexLoadError: 검증 실패 시 (startup fail-fast)
    """
    # 1. FAISS 타입 및 차원
    if not isinstance(index, faiss.IndexFlatL2):
        logger.warning("Index is not IndexFlatL2 (got %s)", type(index))
        # 계속 진행하되 경고 (미래에 다른 인덱스 지원 시)

    expected_dim = settings.embedding_dim
    if index.d != expected_dim:
        raise IndexLoadError(
            f"FAISS index dimension mismatch: expected {expected_dim}, got {index.d}. "
            "Rebuild index with matching embedding model."
        )

    # 2. 벡터 수 vs metadata 수
    ntotal = index.ntotal
    nmeta = len(metadata)
    if ntotal == 0:
        raise IndexLoadError("FAISS index is empty (ntotal=0).")
    if nmeta == 0:
        raise IndexLoadError("Metadata is empty.")
    if ntotal != nmeta:
        raise IndexLoadError(f"Vector count mismatch: index.ntotal={ntotal} != metadata len={nmeta}")

    # 3. 메타데이터 샘플 키 검증 (첫 번째 아이템)
    first = metadata[0]
    if not isinstance(first, dict):
        raise IndexLoadError(f"metadata[0] must be dict, got {type(first)}")

    missing = REQUIRED_META_KEYS - first.keys()
    if missing:
        raise IndexLoadError(f"metadata[0] missing required keys: {missing}. Present keys: {list(first.keys())}")

    logger.info(
        "Index validation passed: ntotal=%d, dim=%d, type=%s",
        ntotal,
        index.d,
        type(index).__name__,
    )


def load_index(index_dir: Path | str | None = None) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """FAISS 인덱스와 metadata를 로드하고 검증.

    lifespan에서 호출되는 주요 진입점. 실패 시 IndexLoadError 발생 (앱 시작 중단).

    Args:
        index_dir: 오버라이드 경로 (None이면 settings.index_dir 사용)

    Returns:
        (faiss.Index, list[dict]) — 검색에 바로 사용 가능

    Raises:
        IndexLoadError: 파일 없음, 형식 오류, 검증 실패 등
    """
    start = time.perf_counter()
    idx_dir = _resolve_index_dir(index_dir)

    faiss_path = idx_dir / "faiss.index"
    if not faiss_path.exists():
        raise IndexLoadError(f"faiss.index not found at {faiss_path}")

    logger.info("Loading vector index from %s ...", idx_dir)

    # 1. Metadata 로드 (pkl 우선)
    metadata = _load_metadata(idx_dir)

    # 2. FAISS 인덱스 로드 (동기, startup 허용)
    try:
        index = faiss.read_index(str(faiss_path))
    except Exception as exc:
        raise IndexLoadError(f"Failed to read faiss.index: {exc}") from exc

    # 3. 검증
    _validate_index_and_metadata(index, metadata)

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "Vector index ready: %d vectors (%.1f ms load time)",
        index.ntotal,
        elapsed,
    )

    return index, metadata


__all__ = ["load_index", "IndexLoadError"]
