"""Tests for the QA sample runner utility (scripts/run_qa_samples.py).

Focus:
- Pure helper functions (load_samples, make_output_record, write_jsonl_record)
- CLI behavior via subprocess (dry-run + error paths) — no network, no server required
- Uses only stdlib + pytest tmp_path; loads the script via importlib.util when needed
  so we do not need to modify the script or add scripts/__init__.py

All tests are fast and self-contained.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _load_runner_module():
    """Dynamically load the runner script without requiring it to be a package."""
    script_path = Path("scripts/run_qa_samples.py").resolve()
    spec = importlib.util.spec_from_file_location("qa_runner", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    """Provide the loaded runner module for helper function tests."""
    return _load_runner_module()


# ------------------------------
# load_samples tests (both supported input structures)
# ------------------------------

def test_load_samples_wrapped_structure(runner, tmp_path):
    data = {
        "samples": [
            {
                "id": "q-001",
                "question": "광주 청년 구직활동수당 신청 방법은?",
                "ground_truth": "광주청년통합플랫폼에서 신청합니다.",
                "difficulty": "easy",
                "qa_type": "factual",
                "category": "employment",
                "policy_title": "구직활동수당",
                "policy_id": "20260420005400212772",
            },
            {"id": "q-002", "question": "두 번째 질문?"},
        ]
    }
    p = tmp_path / "wrapped.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    samples = runner.load_samples(p)
    assert len(samples) == 2
    assert samples[0]["id"] == "q-001"
    assert samples[0]["difficulty"] == "easy"
    # raw dict from JSON does not contain the key; make_output_record turns missing fields into None
    assert "ground_truth" not in samples[1]


def test_load_samples_direct_list_structure(runner, tmp_path):
    data = [
        {"id": "l-01", "question": "리스트 형식 첫 번째"},
        {"id": "l-02", "question": "두 번째", "ground_truth": "참고 답"},
    ]
    p = tmp_path / "list.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    samples = runner.load_samples(p)
    assert len(samples) == 2
    assert samples[0]["question"] == "리스트 형식 첫 번째"


def test_load_samples_errors(runner, tmp_path):
    # dict without "samples"
    p = tmp_path / "bad_dict.json"
    p.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        runner.load_samples(p)
    assert "no 'samples' key" in str(exc.value)

    # samples not a list
    p2 = tmp_path / "bad_samples.json"
    p2.write_text(json.dumps({"samples": "not a list"}), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        runner.load_samples(p2)
    assert "'samples' must be a list" in str(exc.value)

    # top level neither dict nor list
    p3 = tmp_path / "bad_top.json"
    p3.write_text(json.dumps("just a string"), encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        runner.load_samples(p3)
    assert "Unsupported JSON top-level type" in str(exc.value)


# ------------------------------
# make_output_record tests
# ------------------------------

def test_make_output_record_full_passthrough(runner):
    sample = {
        "id": "q-007",
        "question": "질문?",
        "ground_truth": "정답",
        "difficulty": "hard",
        "qa_type": "reasoning",
        "category": "housing",
        "policy_title": "주거 지원",
        "policy_id": "pid-999",
    }
    rec = runner.make_output_record(sample, "http://localhost:8080/api/v1/ask", 5)
    # Must contain exactly the 16 keys in the documented order
    expected_keys = [
        "id", "question", "ground_truth", "difficulty", "qa_type", "category",
        "policy_title", "policy_id",
        "answer", "contexts", "sources", "model", "latency_ms",
        "top_k", "api_url", "error",
    ]
    assert list(rec.keys()) == expected_keys
    assert rec["id"] == "q-007"
    assert rec["difficulty"] == "hard"
    assert rec["answer"] is None
    assert rec["error"] is None
    assert rec["top_k"] == 5
    assert rec["api_url"] == "http://localhost:8080/api/v1/ask"


def test_make_output_record_missing_fields_become_null(runner):
    sample = {"id": "minimal", "question": "짧은 질문"}
    rec = runner.make_output_record(sample, "http://ex", 3)
    assert rec["ground_truth"] is None
    assert rec["difficulty"] is None
    assert rec["policy_title"] is None
    assert rec["answer"] is None
    assert rec["contexts"] is None


def test_make_output_record_non_dict_sample(runner):
    rec = runner.make_output_record("not a dict", "http://ex", 1)
    assert rec["id"] is None
    assert rec["question"] is None
    assert rec["error"] is None  # caller is responsible for setting error


# ------------------------------
# write_jsonl_record (basic)
# ------------------------------

def test_write_jsonl_record_writes_lines(runner, tmp_path):
    out = tmp_path / "results.jsonl"
    rec1 = {"id": "1", "question": "q1", "answer": None, "error": None}
    rec2 = {"id": "2", "question": "q2", "answer": "a2", "error": None}

    with out.open("w", encoding="utf-8") as f:
        runner.write_jsonl_record(f, rec1)
        runner.write_jsonl_record(f, rec2)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "1"
    assert json.loads(lines[1])["answer"] == "a2"


# ------------------------------
# End-to-end CLI tests via subprocess (dry-run + error paths)
# ------------------------------

def test_cli_dry_run_basic(tmp_path):
    """Run the script with --dry-run using a wrapped input. No server needed."""
    input_file = tmp_path / "qa_pairs.json"
    input_file.write_text(
        json.dumps(
            {
                "samples": [
                    {"id": "d-1", "question": "첫 번째 질문입니다."},
                    {"id": "d-2", "question": "두 번째 질문?", "ground_truth": "예상 답변"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    output_file = tmp_path / "rag_eval_results.jsonl"

    project_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/run_qa_samples.py",
        "--dry-run",
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--limit",
        "2",
        "--top-k",
        "3",
        "--verbose",
    ]

    proc = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, f"stderr: {proc.stderr}"
    assert "DRY-RUN" in proc.stdout
    assert "Done. Processed: 2" in proc.stdout

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    rec = json.loads(lines[0])
    assert rec["id"] == "d-1"
    assert rec["question"] == "첫 번째 질문입니다."
    assert rec["answer"] is None
    assert rec["contexts"] is None
    assert rec["error"] is None
    assert rec["top_k"] == 3
    assert rec["api_url"] == "http://localhost:8080/api/v1/ask"


def test_cli_dry_run_direct_list_input(tmp_path):
    input_file = tmp_path / "list_input.json"
    input_file.write_text(
        json.dumps([{"id": "l1", "question": "리스트로 된 질문"}], ensure_ascii=False),
        encoding="utf-8",
    )
    output_file = tmp_path / "out.jsonl"

    project_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_qa_samples.py",
            "--dry-run",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    data = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert data["id"] == "l1"


def test_cli_bad_input_structure_exits_nonzero(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "samples"}), encoding="utf-8")

    project_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_qa_samples.py",
            "--dry-run",
            "--input",
            str(bad),
            "--output",
            str(tmp_path / "ignored.jsonl"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "no 'samples' key" in proc.stderr or "Failed to load samples" in proc.stderr


def test_cli_missing_input_file_exits_nonzero(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_qa_samples.py",
            "--dry-run",
            "--input",
            str(tmp_path / "does_not_exist.json"),
            "--output",
            str(tmp_path / "out.jsonl"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "Input file not found" in proc.stderr


def test_cli_error_recording_with_bogus_url(tmp_path):
    """Exercise the non-dry-run error path (call_ask) using an unroutable URL.
    This verifies error records are still written with input fields preserved.
    """
    input_file = tmp_path / "in.json"
    input_file.write_text(
        json.dumps({"samples": [{"id": "err-1", "question": "실패할 질문"}]}),
        encoding="utf-8",
    )
    output_file = tmp_path / "errors.jsonl"

    project_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_qa_samples.py",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--api-url",
            "http://127.0.0.1:1/api/v1/ask",  # guaranteed to fail fast
            "--timeout",
            "2",
            "--verbose",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # The script itself should still succeed (it continues on per-sample errors)
    assert proc.returncode == 0
    assert output_file.exists()

    rec = json.loads(output_file.read_text(encoding="utf-8").strip())
    assert rec["id"] == "err-1"
    assert rec["question"] == "실패할 질문"
    assert rec["answer"] is None
    assert rec["error"] is not None
    assert "Connection" in rec["error"] or "HTTP" in rec["error"] or "Timeout" in rec["error"].lower()
    assert rec["top_k"] == 5  # default
    assert "127.0.0.1:1" in rec["api_url"]
