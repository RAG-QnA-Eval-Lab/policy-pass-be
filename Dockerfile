# syntax=docker/dockerfile:1.7
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --disable-pip-version-check ".[rag]"
COPY config/ config/
COPY src/ src/
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
