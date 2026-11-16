# Production RAG System — evaluation corpus
#
# This is the ONLY knowledge source for the eval suite (data/corpus.md).
# Sources referenced by evals/test_set.py: rls, agent-safety, evals,
# chunking, retrieval, embeddings, citations, deployment.

## Row-Level Security (rls)
Row-level security restricts data access per user. In a multi-tenant system, every query carries the caller's tenant policy and the database enforces it on rows. A user can never read or write rows outside their own scope, even if application code is bypassed.

## Agent safety model (agent-safety)
The agent executes only allow-listed tools with schema-validated arguments. Before any write action reaches a system, the agent stops and waits for a named human approval. Duplicate requests never double-execute because every run carries an idempotency key. Every action is appended to a hash-chained audit ledger, so tampering is detectable.

## Eval regression gate (evals)
Every model or retrieval change must pass a McNemar paired test against the golden eval set. If quality does not improve with statistical significance, CI fails and the change is blocked. The golden set is fixed and versioned so regression is measurable across commits.

## Chunking (chunking)
Documents are split on paragraph boundaries first, then sentence boundaries, so no chunk breaks mid-sentence. Chunks that still exceed the budget are split further at sentence boundaries. Each chunk stores its source, index, and token count.

## Hybrid retrieval (retrieval)
Retrieval fuses dense embeddings with BM25 keyword scores using Reciprocal Rank Fusion (RRF). A lightweight reranker scores candidates by term overlap. The final score weights RRF (0.5), rerank overlap (0.3), and dense cosine (0.2).

## Embeddings (embeddings)
The default embedding model is text-embedding-3-small. When no API key is set, a deterministic local hash-based fallback is used so the pipeline runs fully offline and in CI.

## Citations and sufficiency (citations)
Every answer is chained to verbatim source quotes with the source named. When no retrieved passage meets the relevance gate, the system returns insufficient_evidence instead of fabricating an answer.

## Deployment (deployment)
The service runs on FastAPI, started with uvicorn (make dev). GitHub Actions runs pytest, lint, and the eval regression gate on every push. The service ships in Docker for one-command deploy.
