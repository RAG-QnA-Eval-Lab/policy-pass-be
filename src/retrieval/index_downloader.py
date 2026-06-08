"""S3 index downloader for production startup.

ensure_index_files():
- If DOWNLOAD_INDEX_FROM_S3=false (default): no-op (local data/index behavior preserved).
- If true and S3_BUCKET empty: raise clear error.
- Downloads faiss.index + metadata.json (required) + metadata.pkl (if present) using boto3 default creds.
- Downloads to a temporary directory first, validates required files (exist + non-empty).
- Then creates target (settings.index_dir) if needed and copies/replaces files there.
- settings.index_dir is NEVER mutated (remains the configured stable target).
- Required files always atomically replaced in target (via .tmp + os.replace where possible).
- Optional metadata.pkl copied only if it was successfully downloaded.
- Keeps local dev unchanged; no AWS creds in code or .env.example.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from config.settings import settings

from .exceptions import IndexLoadError

logger = logging.getLogger(__name__)


def ensure_index_files() -> None:
    """Download FAISS index files from S3 if DOWNLOAD_INDEX_FROM_S3=true.

    Safe for local dev (skips when flag false). Never mutates settings.index_dir.
    Uses settings.index_dir only as the final stable target directory.
    """
    if not settings.download_index_from_s3:
        logger.debug("DOWNLOAD_INDEX_FROM_S3=false; skipping S3 index download (using local index_dir)")
        return

    bucket = (settings.s3_bucket or "").strip()
    if not bucket:
        raise RuntimeError(
            "DOWNLOAD_INDEX_FROM_S3=true but S3_BUCKET is not set. "
            "Set S3_BUCKET and (optionally) INDEX_S3_PREFIX, or set DOWNLOAD_INDEX_FROM_S3=false for local dev."
        )

    prefix = (settings.index_s3_prefix or "index/").strip()
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    s3 = boto3.client("s3")

    required_files = ["faiss.index", "metadata.json"]
    optional_files = ["metadata.pkl"]

    tmp_dir = Path(tempfile.mkdtemp(prefix="policy-pass-index-"))
    logger.info(
        "Downloading index from s3://%s/%s to temp %s (target will be %s)",
        bucket,
        prefix,
        tmp_dir,
        settings.index_dir,
    )

    try:
        for fname in required_files + optional_files:
            key = f"{prefix}{fname}"
            dest = tmp_dir / fname
            try:
                s3.download_file(bucket, key, str(dest))
                if dest.exists() and dest.stat().st_size > 0:
                    logger.debug("Downloaded %s (%d bytes)", fname, dest.stat().st_size)
            except ClientError as ce:
                code = ce.response.get("Error", {}).get("Code", "")
                if fname in optional_files and code in ("404", "NoSuchKey", "NotFound"):
                    logger.info("%s not present in S3 (optional); skipping", fname)
                    continue
                # re-raise for required or unexpected
                logger.error("Failed to download %s from S3: %s", fname, ce)
                raise
            except Exception as exc:
                if fname in optional_files:
                    logger.info("Optional %s download skipped: %s", fname, exc)
                    continue
                raise

        # Validate required files in temp dir
        for fname in required_files:
            p = tmp_dir / fname
            if not p.exists():
                raise IndexLoadError(f"Required file {fname} missing after S3 download from {bucket}/{prefix}")
            if p.stat().st_size == 0:
                raise IndexLoadError(f"Required file {fname} is empty after S3 download")

        # Prepare final target (do NOT mutate settings.index_dir)
        target_dir = Path(settings.index_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        # For required files: copy to .tmp then os.replace for atomic replace-if-possible
        for fname in required_files:
            src = tmp_dir / fname
            dst = target_dir / fname
            dst_tmp = target_dir / (fname + ".tmp")
            shutil.copy2(src, dst_tmp)
            os.replace(dst_tmp, dst)
            logger.debug("Atomically replaced required file in target: %s", dst)

        # Optional metadata.pkl: copy ONLY if it was downloaded (present+nonempty in temp)
        pkl_src = tmp_dir / "metadata.pkl"
        if pkl_src.exists() and pkl_src.stat().st_size > 0:
            pkl_dst = target_dir / "metadata.pkl"
            pkl_tmp = target_dir / "metadata.pkl.tmp"
            shutil.copy2(pkl_src, pkl_tmp)
            os.replace(pkl_tmp, pkl_dst)
            logger.debug("Copied optional metadata.pkl into target: %s", pkl_dst)

        # Success: cleanup temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info(
            "S3 index download complete and validated. Files placed into configured index_dir=%s",
            target_dir,
        )

    except NoCredentialsError as exc:
        # Clear guidance for credential chain issues
        # cleanup partial tmp if present
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            "AWS credentials not available for S3 index download (boto3 default chain). "
            "Configure IAM role, env vars (AWS_ACCESS_KEY_ID etc), or ~/.aws/credentials."
        ) from exc
    except Exception as exc:
        # Best effort cleanup of the temp dir on failure
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        if isinstance(exc, (RuntimeError, IndexLoadError)):
            raise
        raise RuntimeError(f"S3 index download/validation failed: {exc}") from exc


__all__ = ["ensure_index_files"]
