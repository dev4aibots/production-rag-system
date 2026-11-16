# Production RAG System

Production-grade Retrieval-Augmented Generation with verified citations, eval regression gating, and CI. Built to be cloned, run, and measured.

| Pillar | Status | Evidence |
|--------|--------|----------|
| **Execution Mode** | Live Full Backend | `make dev` runs FastAPI + eval suite |
| **Tests** | 18 passing | `pytest` |
| **Eval Gate** | CI blocks regressions | `make eval` + McNemar gate |
| **Authorship** | Solo Original | Devraj Gadhvi |
| **Benchmarks** | Measured locally | `evals/results.jsonl` |

## What it does

A RAG pipeline over a document corpus with:
- **Structure-aware chunking** (respects document boundaries, metadata carried per chunk)
- **Hybrid retrieval** (dense embeddings + BM25 keyword, fused with RRF)
- **Cross-encoder reranking** on top-k candidates
- **Citation enforcement** — every answer chained to verbatim source quotes; unsupported claims return `insufficient_evidence`
- **Eval regression gate** — paired statistical test (McNemar) fails CI on real drops
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
Document → Chunk → Embed → Vector Store (FAISS)
                    ↑
Query → Embed → Hybrid Search (dense + BM25 + RRF) → Rerank → Answer with Citations
```

## Eval Results (measured locally)

| Metric | Value | Provenance | Command |
|--------|-------|------------|---------|
| Recall@5 | 0.91 | Independently measured | `make eval` |
| Citation correctness | 0.94 | Independently measured | `make eval` |
| Faithfulness | 0.89 | Independently measured | `make eval` |
| P95 latency | 2.4s | Independently measured | `make eval` |
| Cost per query | $0.004 | Independently measured | `make eval` |

Raw results committed to `evals/results.jsonl`.

## CI

GitHub Actions runs `pytest` + `make eval` on every push. The eval gate uses a McNemar paired test — it exits non-zero when a change causes a statistically significant regression, so bad code never deploys.

## License

MIT
