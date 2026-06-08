# Policy Pass — Backend (FastAPI)

청년정책 RAG QA 시스템의 FastAPI 백엔드입니다.

## 프로젝트 구조

```
policy-pass-be/
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI 엔트리포인트 + lifespan (startup init)
│   │   ├── chain.py         # LangChain RAG answer chain (build_rag_answer_chain)
│   │   └── routes/
│   │       └── rag.py       # POST /api/v1/search, /api/v1/ask
│   ├── retrieval/           # FAISS in-memory retrieval (embedder, index_loader, retriever, index_downloader)
│   ├── generation/          # Answer generation wrapper (OpenAIGenerator using LangChain)
│   ├── services/            # RAGService orchestration (retriever + generator)
│   └── schemas/             # Pydantic request/response models (stable API contract)
├── config/
│   └── settings.py          # 환경변수 설정 (DOWNLOAD_INDEX_FROM_S3, S3_BUCKET 등)
├── data/index/              # Local FAISS index (faiss.index + metadata.*) for dev
├── tests/
├── scripts/run_qa_samples.py  # QA evaluation runner (calls /ask, writes JSONL)
├── Dockerfile               # 컨테이너 빌드 (포트 8080)
├── pyproject.toml            # 의존성 관리 (rag extras: langchain, langchain-openai, faiss-cpu, boto3)
├── .env.example              # 환경변수 템플릿
└── .github/workflows/
    └── deploy.yml            # CI/CD (ECR push → App Runner 배포)
```

## 로컬 개발 환경 설정

### 1. Python 환경 구성

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,rag]"
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일을 열고 아래 값들을 채워넣으세요
```

| 변수 | 설명 | 필수 |
|------|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (임베딩 검색에 사용) | O |
| `MONGODB_URI` | MongoDB 연결 문자열 | O |
| `MONGODB_DB` | MongoDB 데이터베이스명 (기본: `rag_youth_policy`) | |
| `S3_BUCKET` | FAISS 인덱스 S3 버킷명 | 배포 시 |
| `INDEX_S3_PREFIX` | S3 내 인덱스 경로 (기본: `index/`) | |
| `DOWNLOAD_INDEX_FROM_S3` | S3에서 인덱스 다운로드 여부 (기본: `false`) | |
| `EMBEDDING_MODEL` | 임베딩 모델 (기본: `openai/text-embedding-3-small`) | |
| `EMBEDDING_DIM` | 임베딩 차원 (기본: `1536`) | |
| `ENVIRONMENT` | 환경 구분: `development` / `production` | |

> **주의**: `EMBEDDING_MODEL`과 `EMBEDDING_DIM`은 GCP에서 빌드한 FAISS 인덱스와 동일해야 합니다. 변경 시 인덱스 리빌드가 필요합니다.

### 3. 서버 실행

```bash
uvicorn src.api.main:app --reload --port 8080
```

### 4. 헬스체크

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

## 개발 가이드

### RAG 체인 구현 (완료된 최종 구조)

- `src/api/chain.py`: `build_rag_answer_chain(llm)` — ChatPromptTemplate + StrOutputParser를 사용한 grounded 한국어 RAG 프롬프트 체인.
- `src/generation/generator.py`: `OpenAIGenerator`가 LangChain `ChatOpenAI` + 위 체인을 래핑 (기존 `generate(question, contexts)` 시그니처 유지).
- `src/retrieval/`: FAISS in-memory (IndexFlatL2) + OpenAI embedder. `VectorSearchRetriever`는 `/search`와 `/ask` 모두에서 재사용.
- `src/retrieval/index_downloader.py`: `ensure_index_files()` — production 시 S3에서 인덱스 다운로드 (local에서는 스킵).
- FastAPI lifespan (`main.py`)에서 S3 다운로드(선택) → index load → embedder/retriever/generator 초기화.

API contract는 안정적으로 유지됩니다 (`/api/v1/search`, `/api/v1/ask`).

### FAISS 인덱스

| 항목 | 값 |
|------|------|
| 임베딩 모델 | `openai/text-embedding-3-small` |
| 차원 | `1536` |

> **인덱스 빌드와 질의(쿼리 임베딩) 시 반드시 동일한 모델/차원을 사용해야 합니다.**
> 다른 모델을 쓰면 차원 불일치 에러 또는 검색 품질 저하가 발생합니다.
> 모델을 변경하려면 GCP 파이프라인에서 인덱스를 리빌드해야 합니다.

- 인덱스는 GCP 파이프라인에서 빌드되어 S3에 저장됩니다
- **로컬 개발**: `data/index/` 디렉토리에 `faiss.index`, `metadata.json` (또는 .pkl) 배치 (git tracked for baseline)
- **배포 / production**: `DOWNLOAD_INDEX_FROM_S3=true` + `S3_BUCKET=...` → lifespan에서 `ensure_index_files()` 호출 후 S3에서 다운로드하여 설정된 INDEX_DIR (data/index 등) 에 파일을 배치 (settings.index_dir 값 자체는 변경되지 않음). 로컬 data/index는 변경되지 않음.

### 새 라이브러리 추가 시

새 패키지를 설치했다면 **반드시 `pyproject.toml`에 추가**해야 배포에 반영됩니다.

- 앱 실행에 필요한 패키지 → `dependencies` 에 추가
- RAG 관련 패키지 → `[project.optional-dependencies]`의 `rag` 에 추가

```toml
# pyproject.toml

dependencies = [
    # ... 기존 패키지들
    "새패키지>=1.0",          # ← 여기에 추가
]

[project.optional-dependencies]
rag = [
    # ... 기존 RAG 패키지들
    "새RAG패키지>=1.0",       # ← RAG 관련이면 여기에 추가
]
```

> Dockerfile은 수정할 필요 없습니다. `pip install ".[rag]"`이 pyproject.toml을 자동으로 읽습니다.

## 배포 전 로컬 Docker 테스트

배포 환경과 동일한 조건에서 테스트하려면 [Docker Desktop](https://www.docker.com/products/docker-desktop/)을 설치하고 아래를 실행하세요:

```bash
docker build -t rag-api .
docker run -p 8080:8080 --env-file .env rag-api
# http://localhost:8080/health → {"status":"ok"} 확인
```

**로컬 data/index 마운트하여 테스트 (추천, 인덱스 복사 없이):**

```bash
docker run -p 8080:8080 --env-file .env -v "$PWD/data/index:/app/data/index:ro" rag-api
# (DOWNLOAD_INDEX_FROM_S3=false 인 경우 로컬 마운트된 인덱스 사용)
```

production-like S3 다운로드 테스트 시에는 `.env`에 `DOWNLOAD_INDEX_FROM_S3=true`와 `S3_BUCKET`을 설정하세요 (자격증명은 boto3 default chain 사용).

## 브랜치 작업 규칙

**main 브랜치에 직접 push하지 마세요.** 반드시 브랜치를 만들어서 작업하고 PR로 머지합니다.

```bash
# 1. 작업 브랜치 생성
git checkout -b feature/내작업이름

# 2. 코드 작성 후 커밋
git add .
git commit -m "feat: 기능 설명"

# 3. 원격에 push
git push origin feature/내작업이름

# 4. GitHub에서 PR 생성 → 리뷰 → main에 머지
```

- 브랜치 이름 예시: `feature/rag-chain`, `fix/api-timeout`, `refactor/retrieval`
- 머지 전 다른 팀원의 변경사항 반영: `git pull origin main`으로 최신 코드를 받은 뒤 작업하세요
- **main에 머지되면 자동 배포됩니다** (CI/CD는 인프라 담당이 관리)

## 주의사항

- **시크릿을 코드에 하드코딩하지 마세요** → `.env` 사용
- **`.env` 파일은 절대 커밋하지 마세요** → `.gitignore`에 포함되어 있습니다

## 관련 레포

| 레포 | 역할 |
|------|------|
| [policy-pass-fe](https://github.com/RAG-QnA-Eval-Lab/policy-pass-fe) | Streamlit 프론트엔드 |
| [policy-pass-infra-aws](https://github.com/RAG-QnA-Eval-Lab/policy-pass-infra-aws) | AWS 인프라 설정 |
| [policy-pass-datapipeline-gcp](https://github.com/RAG-QnA-Eval-Lab/policy-pass-datapipeline-gcp) | GCP 데이터 파이프라인 |
