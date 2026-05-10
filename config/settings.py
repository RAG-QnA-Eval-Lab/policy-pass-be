"""프로젝트 설정 — pydantic-settings 기반, SSM Parameter Store 또는 .env 로드."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "rag_youth_policy"
    s3_bucket: str = ""
    index_s3_prefix: str = "index/"
    download_index_from_s3: bool = False
    embedding_model: str = "openai/text-embedding-3-small"
    embedding_dim: int = 1536
    environment: str = "development"
    top_k: int = 10
    rerank_top_k: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
