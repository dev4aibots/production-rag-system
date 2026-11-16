# Production RAG System

Production-grade Retrieval-Augmented Generation with verified citations, eval regression gating, and CI. Built to be cloned, run, and measured.

| Pillar | Status | Evidence |
|--------|--------|----------|
| **Execution Mode** | Live Full Backend | `make dev` runs FastAPI + eval suite |
| **Tests** | 17 passing | `pytest` |
| **Evals** | 30-case golden set, results committed | `python evals/run_evals.py` |
| **Authorship** | Solo Original | Devraj Gadhvi |
## What it does

A RAG pipeline over a document corpus with:
- **Structure-aware chunking** (respects document boundaries, metadata carried per chunk)
- **Hybrid retrieval** (dense embeddings + BM25 keyword, fused with RRF)
- **Cross-encoder reranking** on top-k candidates
- **Citation enforcement** â€” every answer chained to verbatim source quotes; unsupported claims return `insufficient_evidence`
- **Eval regression gate** â€” paired statistical test (McNemar) fails CI on real drops
- **FastAPI `/ask` endpoint** with structured Pydantic outputs
- **Docker** one-command deploy

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/dev4aibots/production-rag-system.git
cd production-rag-system
cp .env.example .env
# Add your OPENAI_API_KEY or point at a local embedding model

# 2. Install and run
pip install -r requirements.txt
make dev          # starts FastAPI on http://localhost:8000

# 3. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does row-level security work?", "top_k": 5}'

# 4. Run evals (regression gate)
make eval         # exits non-zero on regression
```

## Architecture

```
Document â†’ Chunk â†’ Embed â†’ Vector Store (FAISS)
                    â†‘
Query â†’ Embed â†’ Hybrid Search (dense + BM25 + RRF) â†’ Rerank â†’ Answer with Citations
```


## Evals

A committed 30-case golden test set (`evals/test_set.py`) over a fixed corpus
(`evals/corpus.md`), with a runner that reports metrics and writes results:

```bash
python evals/run_evals.py            # offline deterministic path
python evals/run_evals.py --llm      # real LLM generation (requires OPENAI_API_KEY)
```

Committed results: `evals/results-summary.json`, per-case `evals/results.jsonl`,
baseline `evals/baseline.json`.

| Metric | Value (offline) |
|--------|------------------|
| Accuracy | 0.933 |
| Answerable correct | 0.913 |
| Unanswerable rejected | 1.000 |
| Citation correctness | 1.000 |
| Faithfulness | 0.933 |
| Recall@k | 0.767 |
| Latency p50 / p95 | 1 / 2 ms |
| Est. cost (LLM path, per run) | $0.003 |

The LLM service (`src/llm_service.py`) enforces a per-IP rate limit and a
daily spend cap; keys are read from environment only.

## CI

GitHub Actions runs `pytest` + `make eval` on every push. The eval gate uses a McNemar paired test â€” it exits non-zero when a change causes a statistically significant regression, so bad code never deploys.

## License

MIT



