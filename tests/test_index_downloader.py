"""Minimal tests for src/retrieval/index_downloader.py.

- No real AWS / boto3 / S3 calls (use mocks).
- No real OpenAI.
- Covers required scenarios + new invariants (no mutation of settings.index_dir,
  files land in configured target, pkl copied only when provided) using monkeypatch + MagicMock.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.retrieval.index_downloader as index_downloader


def _make_fake_settings(**overrides):
    """Return a simple object with the attrs used by ensure_index_files."""
    base = {
        "download_index_from_s3": False,
        "s3_bucket": "",
        "index_s3_prefix": "index/",
        "index_dir": "data/index",
    }
    base.update(overrides)
    return type("FakeSettings", (), base)()


def test_ensure_skipped_when_download_false(monkeypatch):
    """1. ensure_index_files() is skipped when DOWNLOAD_INDEX_FROM_S3=false."""
    fake_settings = _make_fake_settings(download_index_from_s3=False)
    monkeypatch.setattr(index_downloader, "settings", fake_settings)

    # Patch boto3 to ensure it is never called
    with patch("src.retrieval.index_downloader.boto3") as mock_boto3:
        index_downloader.ensure_index_files()
        mock_boto3.client.assert_not_called()

    # index_dir should remain unchanged (the fake one)
    assert fake_settings.index_dir == "data/index"


def test_ensure_raises_when_true_and_bucket_empty(monkeypatch):
    """2. ensure_index_files() raises a clear error when DOWNLOAD=true and S3_BUCKET empty."""
    fake_settings = _make_fake_settings(
        download_index_from_s3=True,
        s3_bucket="",  # empty
        index_s3_prefix="index/",
    )
    monkeypatch.setattr(index_downloader, "settings", fake_settings)

    with patch("src.retrieval.index_downloader.boto3") as mock_boto3:
        with pytest.raises(RuntimeError) as exc:
            index_downloader.ensure_index_files()
        msg = str(exc.value)
        assert "DOWNLOAD_INDEX_FROM_S3=true" in msg
        assert "S3_BUCKET" in msg
        mock_boto3.client.assert_not_called()


def test_ensure_downloads_and_validates_required_files(monkeypatch, tmp_path):
    """3. Required files can be downloaded and validated using a fake S3 client.

    The fake download_file writes non-empty content to the dest passed by the downloader.
    After success:
    - settings.index_dir must NOT be mutated (remains the configured target).
    - The (stable) target dir now contains the required files (copied from temp).
    - metadata.pkl is NOT present in target because fake S3 did not provide it.
    """
    original_index_dir = str(tmp_path / "original")
    fake_settings = _make_fake_settings(
        download_index_from_s3=True,
        s3_bucket="my-test-bucket",
        index_s3_prefix="index/",
        index_dir=original_index_dir,  # must remain stable; files will be placed here
    )
    monkeypatch.setattr(index_downloader, "settings", fake_settings)

    # Prepare content that the mock will "download"
    faiss_content = b"fake-faiss-bytes-12345"
    meta_json_content = b'[{"policy_id":"p1","title":"t1","content":"c1"}]'

    def fake_download_file(bucket, key, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("faiss.index"):
            dest.write_bytes(faiss_content)
        elif key.endswith("metadata.json"):
            dest.write_bytes(meta_json_content)
        elif key.endswith("metadata.pkl"):
            # optional, write or not — downloader should tolerate absence
            pass  # do not write; treat as not present in S3 for this test
        else:
            # unknown key — do nothing (will cause validation fail if required)
            pass

    fake_s3 = MagicMock()
    fake_s3.download_file.side_effect = fake_download_file

    with patch("src.retrieval.index_downloader.boto3.client", return_value=fake_s3) as mock_client:
        # Call under test
        index_downloader.ensure_index_files()

        # boto3.client was called for s3
        mock_client.assert_called_once_with("s3")

        # Two required + we simulated optional not present
        assert fake_s3.download_file.call_count >= 2

        # CRITICAL: settings.index_dir must NOT have been mutated
        assert fake_settings.index_dir == original_index_dir

        # Files must exist in the *configured target dir* (not a mutated tmp)
        target_dir = Path(fake_settings.index_dir)
        assert target_dir.exists()
        assert (target_dir / "faiss.index").exists()
        assert (target_dir / "faiss.index").stat().st_size > 0
        assert (target_dir / "metadata.json").exists()
        assert (target_dir / "metadata.json").stat().st_size > 0

        # optional pkl was not provided by fake S3 -> must not be copied to target
        assert not (target_dir / "metadata.pkl").exists()


def test_ensure_raises_on_missing_required_after_download(monkeypatch):
    """Edge: if S3 'succeeds' but a required file ends up empty/missing, validation fails."""
    fake_settings = _make_fake_settings(
        download_index_from_s3=True,
        s3_bucket="b",
        index_s3_prefix="idx/",
    )
    monkeypatch.setattr(index_downloader, "settings", fake_settings)

    def fake_download_incomplete(bucket, key, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("faiss.index"):
            dest.write_bytes(b"ok")
        # deliberately omit metadata.json

    fake_s3 = MagicMock()
    fake_s3.download_file.side_effect = fake_download_incomplete

    with patch("src.retrieval.index_downloader.boto3.client", return_value=fake_s3):
        with pytest.raises(Exception) as exc:  # IndexLoadError or RuntimeError
            index_downloader.ensure_index_files()
        assert "metadata.json" in str(exc.value) or "missing" in str(exc.value).lower()


def test_ensure_copies_metadata_pkl_only_if_provided_by_s3(monkeypatch, tmp_path):
    """Verify optional metadata.pkl is copied to target ONLY when fake S3 actually provides it.

    Also re-asserts that settings.index_dir is never mutated.
    """
    original_index_dir = str(tmp_path / "target-with-pkl")
    fake_settings = _make_fake_settings(
        download_index_from_s3=True,
        s3_bucket="my-bucket",
        index_s3_prefix="idx/",
        index_dir=original_index_dir,
    )
    monkeypatch.setattr(index_downloader, "settings", fake_settings)

    faiss_content = b"faiss-bytes"
    meta_json_content = b'[{"policy_id":"p2"}]'
    pkl_content = b"pickle-bytes-for-metadata"

    def fake_download_file(bucket, key, dest_path):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if key.endswith("faiss.index"):
            dest.write_bytes(faiss_content)
        elif key.endswith("metadata.json"):
            dest.write_bytes(meta_json_content)
        elif key.endswith("metadata.pkl"):
            dest.write_bytes(pkl_content)  # this time S3 provides it
        # no else needed

    fake_s3 = MagicMock()
    fake_s3.download_file.side_effect = fake_download_file

    with patch("src.retrieval.index_downloader.boto3.client", return_value=fake_s3):
        index_downloader.ensure_index_files()

        # still must not mutate
        assert fake_settings.index_dir == original_index_dir

        target_dir = Path(fake_settings.index_dir)
        assert (target_dir / "faiss.index").exists()
        assert (target_dir / "metadata.json").exists()
        # now pkl MUST be copied because S3 provided it
        assert (target_dir / "metadata.pkl").exists()
        assert (target_dir / "metadata.pkl").stat().st_size > 0
        assert (target_dir / "metadata.pkl").read_bytes() == pkl_content
