# Policy Pass — Backend (FastAPI)

청년정책 RAG QA 시스템의 FastAPI 백엔드입니다.

## 프로젝트 구조

```
policy-pass-be/
├── src/
│   └── api/
│       └── main.py          # FastAPI 엔트리포인트
├── config/
│   └── settings.py          # 환경변수 설정
├── tests/
├── Dockerfile               # 컨테이너 빌드 (포트 8080)
├── pyproject.toml            # 의존성 관리
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

### RAG 체인 구현

`src/api/` 디렉토리에 LangChain 기반 RAG 로직을 추가하세요:

```
src/
├── api/
│   ├── main.py           # FastAPI 앱
│   ├── routes/           # API 라우트 (/ask, /models 등)
│   └── chain.py          # LangChain RAG 체인
├── retrieval/            # FAISS 검색 로직
└── generation/           # LLM 답변 생성
```

### FAISS 인덱스

- 인덱스는 GCP 파이프라인에서 빌드되어 S3에 저장됩니다
- **로컬 개발**: `data/index/` 디렉토리에 `faiss.index`, `metadata.json` 배치
- **배포**: `DOWNLOAD_INDEX_FROM_S3=true` → 앱 시작 시 S3에서 자동 다운로드

### 테스트

```bash
pytest
```

## 배포 (CI/CD)

### Step 1: Daehyun(인프라 담당)에게 받아야 할 것

| 항목 | 설명 |
|------|------|
| AWS Secret Access Key | `toby` IAM 사용자의 Secret Key (대면 전달) |
| App Runner 서비스 ARN | `rag-qa-api` 서비스의 ARN |
| MongoDB 접속 정보 | URI, DB명, 방화벽 상태 |

### Step 2: AWS CLI 프로필 설정

```bash
aws configure --profile rag-qa
# AWS Access Key ID: Daehyun에게 문의
# AWS Secret Access Key: Daehyun에게 문의
# Default region name: ap-northeast-2
# Default output format: json
```

검증:
```bash
aws sts get-caller-identity --profile rag-qa
# Account: 355206939988 이 나와야 함
```

### Step 3: GitHub Secrets 등록

레포 → **Settings** → **Secrets and variables** → **Actions**:

| Secret Name | 값 | 설명 |
|-------------|-----|------|
| `AWS_ACCESS_KEY_ID` | Daehyun에게 문의 | AWS IAM Access Key |
| `AWS_SECRET_ACCESS_KEY` | Daehyun에게 문의 | AWS IAM Secret Key |
| `APPRUNNER_SERVICE_ARN` | Daehyun에게 문의 | App Runner API 서비스 ARN |

### Step 4: 로컬 Docker 빌드 테스트

```bash
docker build -t rag-api .
docker run -p 8080:8080 --env-file .env rag-api
# http://localhost:8080/health 접속 → {"status":"ok"}
```

### 배포 흐름

```
main 브랜치에 push (또는 수동 트리거)
    ↓
GitHub Actions 실행
    ↓
Docker 이미지 빌드
    ↓
ECR (rag-api)에 push
    ↓
App Runner 자동 재배포
```

- **main 브랜치에 push하면 자동 배포됩니다**
- PR을 먼저 만들고 리뷰 후 머지하세요
- 수동 배포: GitHub → Actions 탭 → Run workflow

## 주의사항

- **인덱스 파일을 Docker 이미지에 포함하지 마세요** → 런타임에 S3에서 다운로드
- **시크릿을 코드에 하드코딩하지 마세요** → `.env` 또는 SSM Parameter Store 사용
- **`.env` 파일은 절대 커밋하지 마세요** → `.gitignore`에 포함되어 있습니다
- MongoDB는 GCP Compute Engine에서 운영 중 → 방화벽 설정 확인 필요

## 관련 레포

| 레포 | 역할 |
|------|------|
| [policy-pass-fe](https://github.com/RAG-QnA-Eval-Lab/policy-pass-fe) | Streamlit 프론트엔드 |
| [policy-pass-infra-aws](https://github.com/RAG-QnA-Eval-Lab/policy-pass-infra-aws) | AWS 인프라 설정 |
| [policy-pass-datapipeline-gcp](https://github.com/RAG-QnA-Eval-Lab/policy-pass-datapipeline-gcp) | GCP 데이터 파이프라인 |
