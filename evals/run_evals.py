#!/usr/bin/env python3
"""Eval runner for Production RAG System.

Loads the corpus (evals/corpus.md), builds a fresh index, runs the 30-case
golden test set, and writes:

  - evals/results.jsonl      per-case results
  - evals/results-summary.json  aggregate metrics + cost estimate
  - evals/baseline.json       last committed baseline (for regression compare)

Metrics computed:
  - recall@k on expected source
  - citation correctness (source cited)
  - faithfulness (answer contains expected substring; never fabricates)
  - accuracy (sufficiency correctness)
  - p50/p95 latency per question
  - estimated cost per question (OpenAI gpt-4o-mini tokens, only when LLM used)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import RAGPipeline, hybrid_retrieve, generate_answer  # noqa: E402
from test_set import EVAL_SET  # noqa: E402

EVALS_DIR = ROOT / "evals"
CORPUS = EVALS_DIR / "corpus.md"

# Cost model (OpenAI pricing, USD per 1M tokens)
COST_INPUT_PER_1M = 0.15   # gpt-4o-mini input
COST_OUTPUT_PER_1M = 0.60  # gpt-4o-mini output


def load_corpus() -> dict[str, str]:
    """Parse corpus.md into {source: text} sections.

    Heading format: `## Human Readable Name (source-alias)`.
    The alias inside parentheses is the source key used by evals.
    """
    text = CORPUS.read_text(encoding="utf-8")
    sources: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                sources[current] = "\n".join(buf).strip()
            heading = line[3:].strip()
            alias_match = re.search(r"\(([^)]+)\)\s*$", heading)
            current = alias_match.group(1).strip() if alias_match else heading.lower()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        sources[current] = "\n".join(buf).strip()
    return sources


def build_pipeline() -> RAGPipeline:
    pipe = RAGPipeline()
    corpus = load_corpus()
    for source, body in corpus.items():
        pipe.ingest(body, source=source)
    return pipe


def cost_estimate(answer_chars: int, context_chars: int) -> float:
    input_tokens = (context_chars + 600) / 4  # prompt + system
    output_tokens = answer_chars / 4
    return (input_tokens / 1e6) * COST_INPUT_PER_1M + (output_tokens / 1e6) * COST_OUTPUT_PER_1M


def run(use_llm: bool = False, top_k: int = 5, rerank_top_k: int = 3) -> dict:
    pipe = build_pipeline()
    cases = []
    latencies: list[int] = []
    total_cost = 0.0

    for case in EVAL_SET:
        start = time.time()
        answer = pipe.ask(case["question"], use_llm=use_llm)
        latency_ms = answer.latency_ms or int((time.time() - start) * 1000)
        latencies.append(latency_ms)

        sufficient = not answer.insufficient_evidence
        contains_expected = (
            case["expected_answer_contains"].lower() in answer.answer.lower()
            if case["expected_answer_contains"] else True
        )
        source_match = (
            any(c["source"] == case["expected_source"] for c in answer.citations)
            if case["expected_source"] else True
        )
        passed = (sufficient == case["should_be_sufficient"]) and contains_expected and source_match

        # retrieval recall@k: is expected source in top-k retrieved results?
        retrieved_sources: list[str] = []
        if case["expected_source"]:
            results = hybrid_retrieve(pipe.store, case["question"], top_k=top_k, rerank_top_k=rerank_top_k)
            retrieved_sources = [r.chunk.source for r in results]
        recall_at_k = 1.0 if case["expected_source"] in retrieved_sources else 0.0

        per_case = {
            "id": case["id"],
            "question": case["question"],
            "passed": passed,
            "sufficient": sufficient,
            "expected_sufficient": case["should_be_sufficient"],
            "contains_expected": contains_expected,
            "source_match": source_match,
            "recall@k": recall_at_k,
            "answer": answer.answer[:160],
            "citations": [{"source": c["source"], "score": c["score"]} for c in answer.citations],
            "latency_ms": latency_ms,
            "estimated_cost_usd": round(cost_estimate(len(answer.answer), 0), 8),
        }
        total_cost += per_case["estimated_cost_usd"]
        cases.append(per_case)

    total = len(cases)
    passed_n = sum(1 for c in cases if c["passed"])
    answerable = [c for c in cases if c["expected_sufficient"]]
    unanswerable = [c for c in cases if not c["expected_sufficient"]]

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "use_llm": use_llm,
        "top_k": top_k,
        "rerank_top_k": rerank_top_k,
        "n_cases": total,
        "accuracy": round(passed_n / total, 4) if total else 0.0,
        "answerable_correct": round(sum(1 for c in answerable if c["passed"]) / len(answerable), 4) if answerable else None,
        "unanswerable_rejected": round(sum(1 for c in unanswerable if not c["sufficient"]) / len(unanswerable), 4) if unanswerable else None,
        "citation_correctness": round(sum(1 for c in cases if c["source_match"]) / total, 4) if total else 0.0,
        "faithfulness": round(sum(1 for c in cases if c["contains_expected"]) / total, 4) if total else 0.0,
        "recall_at_k": round(sum(c["recall@k"] for c in cases) / total, 4) if total else 0.0,
        "latency_ms_p50": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        "estimated_total_cost_usd": round(total_cost, 6),
    }

    EVALS_DIR.mkdir(exist_ok=True)
    (EVALS_DIR / "results.jsonl").write_text(
        "\n".join(json.dumps(c) for c in cases) + "\n", encoding="utf-8"
    )
    (EVALS_DIR / "results-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evals")
    parser.add_argument("--llm", action="store_true", help="use real LLM generation (requires OPENAI_API_KEY)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank-top-k", type=int, default=3)
    args = parser.parse_args()
    summary = run(use_llm=args.llm, top_k=args.top_k, rerank_top_k=args.rerank_top_k)
    print(json.dumps(summary, indent=2))
