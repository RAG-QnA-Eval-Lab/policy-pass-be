#!/usr/bin/env python3
"""Standalone QA sample runner for RAG evaluation.

Reads QA samples from data/eval/qa_pairs.json (or custom --input) and calls the
existing POST /api/v1/ask endpoint for each sample question.

Input file support (exactly as specified):
- Preferred structure (dict with "samples" key):
  {
    "samples": [
      {
        "id": "...",
        "question": "...",
        "ground_truth": "...",
        "difficulty": "...",
        "qa_type": "...",
        "category": "...",
        "policy_title": "...",
        "policy_id": "..."
      }
    ]
  }
- Fallback structure (direct list):
  [
    { "id": "...", "question": "...", "ground_truth": "..." }
  ]

Loading rule:
  1. Load JSON.
  2. If dict and has "samples" key -> use qa_data["samples"].
  3. If list -> use directly.
  4. Otherwise -> fail with clear error.

The script is deliberately standalone:
- Uses only Python standard library (argparse, json, urllib.request, pathlib, etc.).
- Calls the live HTTP API (server must be running with index loaded).
- Never imports from src/ or config/.

Output: JSONL (one object per line) written to --output (default outputs/rag_eval_results.jsonl).
Each record contains exactly these keys (input fields are passed through; missing rich fields become null):

  id, question, ground_truth, difficulty, qa_type, category,
  policy_title, policy_id,
  answer, contexts, sources, model, latency_ms,
  top_k, api_url, error

On errors per sample: the record is still written with "error" populated and
RAG result fields null so that long runs can continue and downstream tools can see partial results.

Usage examples:
  python scripts/run_qa_samples.py --dry-run --verbose
  python scripts/run_qa_samples.py --limit 5 --delay 0.5 --verbose
  python scripts/run_qa_samples.py --api-url http://localhost:8080/api/v1/ask --top-k 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load QA samples supporting both the preferred wrapped structure and the list fallback."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "samples" in data:
            samples = data["samples"]
        else:
            raise ValueError(
                "Input JSON is an object/dict but has no 'samples' key. "
                "Supported formats:\n"
                "  1) {\"samples\": [ {...}, ... ]}\n"
                "  2) [ {...}, ... ] (direct list)\n"
                f"Top-level keys present: {list(data.keys())}"
            )
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(
            f"Unsupported JSON top-level type: {type(data).__name__}. "
            "Expected a dict containing 'samples' or a direct list of sample objects."
        )

    if not isinstance(samples, list):
        raise ValueError(f"'samples' must be a list, got {type(samples).__name__}")

    return samples


def call_ask(api_url: str, question: str, top_k: int, timeout: float) -> dict[str, Any]:
    """POST to the ask endpoint and return the parsed JSON response."""
    payload = {"question": question, "top_k": top_k}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def make_output_record(
    sample: dict[str, Any],
    api_url: str,
    top_k: int,
    answer: Any = None,
    contexts: Any = None,
    sources: Any = None,
    model: Any = None,
    latency_ms: Any = None,
    error: Any = None,
) -> dict[str, Any]:
    """Build a record with the exact 16 keys in the required order."""
    return {
        "id": sample.get("id") if isinstance(sample, dict) else None,
        "question": sample.get("question") if isinstance(sample, dict) else None,
        "ground_truth": sample.get("ground_truth") if isinstance(sample, dict) else None,
        "difficulty": sample.get("difficulty") if isinstance(sample, dict) else None,
        "qa_type": sample.get("qa_type") if isinstance(sample, dict) else None,
        "category": sample.get("category") if isinstance(sample, dict) else None,
        "policy_title": sample.get("policy_title") if isinstance(sample, dict) else None,
        "policy_id": sample.get("policy_id") if isinstance(sample, dict) else None,
        "answer": answer,
        "contexts": contexts,
        "sources": sources,
        "model": model,
        "latency_ms": latency_ms,
        "top_k": top_k,
        "api_url": api_url,
        "error": error,
    }


def write_jsonl_record(out_f, record: dict[str, Any]) -> None:
    """Write one JSON object as a single line and flush (for partial result durability)."""
    line = json.dumps(record, ensure_ascii=False)
    out_f.write(line + "\n")
    out_f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QA samples against POST /api/v1/ask and emit JSONL for RAGAS/DeepEval."
    )
    parser.add_argument(
        "--input",
        default="data/eval/qa_pairs.json",
        help="Path to QA samples JSON file (default: data/eval/qa_pairs.json)",
    )
    parser.add_argument(
        "--output",
        default="outputs/rag_eval_results.jsonl",
        help="Path to write JSONL results (default: outputs/rag_eval_results.jsonl)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8080/api/v1/ask",
        help="Full URL of the ask endpoint (default: http://localhost:8080/api/v1/ask)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N samples (optional)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="top_k value to send with each request (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Per-request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after each sample (default: 0.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without making any HTTP calls",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each sample",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    api_url = args.api_url
    top_k = args.top_k
    timeout = args.timeout
    delay = args.delay
    limit = args.limit
    dry_run = args.dry_run
    verbose = args.verbose

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        samples: list[dict[str, Any]] = load_samples(input_path)
    except Exception as exc:
        print(f"ERROR: Failed to load samples from {input_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if limit is not None and limit > 0:
        samples = samples[:limit]

    total = len(samples)
    if total == 0:
        print("No samples to process after applying --limit (if any).")
        return

    # Ensure output parent directory exists (e.g. outputs/)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    succeeded = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(samples, 1):
            # Basic guard: every sample should be a dict
            if not isinstance(sample, dict):
                record = make_output_record(
                    {},
                    api_url=api_url,
                    top_k=top_k,
                    error=f"Sample #{idx} is not an object/dict",
                )
                write_jsonl_record(out_f, record)
                failed += 1
                processed += 1
                if verbose:
                    print(f"[{idx}/{total}] ERROR: sample is not a dict")
                if delay > 0:
                    time.sleep(delay)
                continue

            question = sample.get("question")

            # Validate presence of question early so we can emit a useful error record
            if not question or not isinstance(question, str):
                record = make_output_record(
                    sample,
                    api_url=api_url,
                    top_k=top_k,
                    error="Missing or invalid 'question' field (must be non-empty string)",
                )
                write_jsonl_record(out_f, record)
                failed += 1
                processed += 1
                if verbose:
                    sid = sample.get("id")
                    print(f"[{idx}/{total}] SKIPPED (no question): id={sid}")
                if delay > 0:
                    time.sleep(delay)
                continue

            if verbose:
                qshort = (question[:60] + "...") if len(question) > 60 else question
                print(f"[{idx}/{total}] Processing: {qshort}")

            answer = None
            contexts = None
            sources = None
            model = None
            latency_ms = None
            error = None

            if dry_run:
                if verbose:
                    print(f"  DRY-RUN -> {api_url} (top_k={top_k})")
            else:
                try:
                    resp = call_ask(api_url, question, top_k, timeout)
                    # Map fields from the AskResponse schema
                    answer = resp.get("answer")
                    contexts = resp.get("contexts")
                    sources = resp.get("sources")
                    model = resp.get("model")
                    latency_ms = resp.get("latency_ms")
                    if verbose:
                        lat_str = f"{latency_ms:.1f}ms" if isinstance(latency_ms, (int, float)) else "?"
                        print(f"  OK (model={model}, latency={lat_str})")
                    succeeded += 1
                except urllib.error.HTTPError as e:
                    # e.g. 502, 500, 404 etc.
                    error = f"HTTP {e.code}: {e.reason}"
                    if verbose:
                        print(f"  ERROR: {error}")
                    failed += 1
                except urllib.error.URLError as e:
                    error = f"Connection error: {e.reason}"
                    if verbose:
                        print(f"  ERROR: {error}")
                    failed += 1
                except TimeoutError as e:
                    error = f"Timeout: {e}"
                    if verbose:
                        print(f"  ERROR: {error}")
                    failed += 1
                except json.JSONDecodeError as e:
                    error = f"Invalid JSON in response: {e}"
                    if verbose:
                        print(f"  ERROR: {error}")
                    failed += 1
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    if verbose:
                        print(f"  ERROR: {error}")
                    failed += 1

            record = make_output_record(
                sample,
                api_url=api_url,
                top_k=top_k,
                answer=answer,
                contexts=contexts,
                sources=sources,
                model=model,
                latency_ms=latency_ms,
                error=error,
            )
            write_jsonl_record(out_f, record)
            processed += 1

            if delay > 0:
                time.sleep(delay)

    print(
        f"\nDone. Processed: {processed}, Succeeded: {succeeded}, Failed: {failed} "
        f"(total samples considered: {total})"
    )
    print(f"Results written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
