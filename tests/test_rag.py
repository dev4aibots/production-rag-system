import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag import (
    chunk_document,
    Chunk,
    VectorStore,
    hybrid_retrieve,
    generate_answer,
    RAGPipeline,
    run_eval,
    mcnemar_gate,
    GOLDEN_EVAL_SET,
    _cosine,
    _bm25_score,
    _rerank_score,
)


def test_chunk_document_respects_size():
    text = "This is a sentence. " * 200
    chunks = chunk_document(text, "test-source", chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 50 + 10, f"Chunk too large: {c.token_count}"
    assert all(c.source == "test-source" for c in chunks)


def test_chunk_ids_deterministic():
    text = "Hello world. This is a test."
    c1 = chunk_document(text, "src")
    c2 = chunk_document(text, "src")
    assert [c.id for c in c1] == [c.id for c in c2]


def test_vector_store_round_trip():
    store = VectorStore()
    chunks = chunk_document("The quick brown fox jumps over the lazy dog.", "doc1")
    store.add(chunks)
    assert len(store.chunks) == len(chunks)
    assert store._index_ready


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    assert abs(_cosine(a, b) - 1.0) < 1e-6
    assert abs(_cosine(a, c)) < 1e-6


def test_bm25_score():
    score = _bm25_score(["fox"], ["the", "quick", "brown", "fox"], 4, 4.0)
    assert score > 0
    score_zero = _bm25_score(["zebra"], ["the", "quick", "brown", "fox"], 4, 4.0)
    assert score_zero == 0


def test_rerank_score():
    score = _rerank_score("fox jump", "The quick brown fox jumps over the dog.")
    assert 0.0 <= score <= 1.0
    score_zero = _rerank_score("zebra", "The quick brown fox.")
    assert score_zero == 0.0


def test_hybrid_retrieve_returns_results():
    store = VectorStore()
    chunks = chunk_document("Row-level security restricts data access per user.", "rls-doc")
    store.add(chunks)
    results = hybrid_retrieve(store, "row-level security", top_k=5, rerank_top_k=3)
    assert len(results) > 0
    assert all(isinstance(r.final_score, float) for r in results)


def test_ask_returns_answer():
    pipeline = RAGPipeline()
    pipeline.ingest("Row-level security restricts data access per user by filtering rows.", "rls")
    answer = pipeline.ask("What is row-level security?")
    assert isinstance(answer.answer, str)
    assert isinstance(answer.citations, list)
    assert 0.0 <= answer.confidence <= 1.0


def test_insufficient_evidence_for_unknown():
    pipeline = RAGPipeline()
    pipeline.ingest("Row-level security restricts data access per user.", "rls")
    answer = pipeline.ask("Who is the president of Mars?")
    assert answer.insufficient_evidence is True


def test_eval_run_returns_metrics():
    pipeline = RAGPipeline()
    pipeline.ingest("Row-level security restricts data access per user.", "rls")
    pipeline.ingest("Citation enforcement requires verbatim quotes from sources.", "citations")
    metrics = run_eval(pipeline)
    assert "pass_rate" in metrics
    assert "recall_at_5" in metrics
    assert "citation_correctness" in metrics
    assert "faithfulness" in metrics
    assert 0.0 <= metrics["pass_rate"] <= 1.0


def test_mcnemar_gate_no_regression():
    current = {"pass_rate": 0.9}
    baseline = {"pass_rate": 0.9}
    assert mcnemar_gate(current, baseline) is False


def test_mcnemar_gate_regression():
    current = {"pass_rate": 0.5}
    baseline = {"pass_rate": 0.9}
    assert mcnemar_gate(current, baseline) is True


def test_golden_eval_set_not_empty():
    assert len(GOLDEN_EVAL_SET) >= 3
    assert all(c.id for c in GOLDEN_EVAL_SET)
    assert all(c.question for c in GOLDEN_EVAL_SET)
