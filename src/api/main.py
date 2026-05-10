"""Policy Pass — FastAPI 백엔드 엔트리포인트."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Policy Pass API starting (env=%s)", settings.environment)
    yield
    logger.info("Policy Pass API shutting down")


app = FastAPI(
    title="Policy Pass API",
    description="청년정책 RAG QA 백엔드",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Policy Pass API", "version": "0.1.0"}
