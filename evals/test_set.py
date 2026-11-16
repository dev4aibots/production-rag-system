# Evals — golden test set for the Production RAG System
#
# Structure per case:
#   id                    stable unique id
#   question              asked of the system
#   expected_answer_contains  substring that must appear in a sufficient answer
#   expected_source        source doc that must be cited
#   should_be_sufficient   True => system must answer; False => must return insufficient_evidence
#
# 30 cases: 24 answerable, 6 unanswerable (must NOT fabricate).
# The corpus (data/corpus.md) is the only knowledge source.

EVAL_SET = [
    # --- answerable: security / RLS ---
    {"id": "e001", "question": "What is row-level security?", "expected_answer_contains": "row-level", "expected_source": "rls", "should_be_sufficient": True},
    {"id": "e002", "question": "How does RLS restrict data access per user?", "expected_answer_contains": "row-level", "expected_source": "rls", "should_be_sufficient": True},
    {"id": "e003", "question": "Where is authorization enforced in the database?", "expected_answer_contains": "query", "expected_source": "rls", "should_be_sufficient": True},
    {"id": "e004", "question": "What happens if application code is bypassed?", "expected_answer_contains": "rows", "expected_source": "rls", "should_be_sufficient": True},
    # --- answerable: agent safety / HITL ---
    {"id": "e005", "question": "What tools can the agent execute?", "expected_answer_contains": "allow-listed", "expected_source": "agent-safety", "should_be_sufficient": True},
    {"id": "e006", "question": "When does the agent stop for human approval?", "expected_answer_contains": "write", "expected_source": "agent-safety", "should_be_sufficient": True},
    {"id": "e007", "question": "How is tampering with the audit ledger detected?", "expected_answer_contains": "hash", "expected_source": "agent-safety", "should_be_sufficient": True},
    {"id": "e008", "question": "What prevents duplicate requests from double-executing?", "expected_answer_contains": "idempotency", "expected_source": "agent-safety", "should_be_sufficient": True},
    # --- answerable: evals / McNemar ---
    {"id": "e009", "question": "What statistical test gates regression in CI?", "expected_answer_contains": "McNemar", "expected_source": "evals", "should_be_sufficient": True},
    {"id": "e010", "question": "What happens when quality drops?", "expected_answer_contains": "CI", "expected_source": "evals", "should_be_sufficient": True},
    {"id": "e011", "question": "What is the golden eval set used for?", "expected_answer_contains": "regression", "expected_source": "evals", "should_be_sufficient": True},
    # --- answerable: chunking ---
    {"id": "e012", "question": "How are large paragraphs split?", "expected_answer_contains": "sentence", "expected_source": "chunking", "should_be_sufficient": True},
    {"id": "e013", "question": "What metadata does each chunk carry?", "expected_answer_contains": "source", "expected_source": "chunking", "should_be_sufficient": True},
    {"id": "e014", "question": "How is a document split first?", "expected_answer_contains": "paragraph", "expected_source": "chunking", "should_be_sufficient": True},
    # --- answerable: hybrid retrieval ---
    {"id": "e015", "question": "Which retrieval signals are fused?", "expected_answer_contains": "BM25", "expected_source": "retrieval", "should_be_sufficient": True},
    {"id": "e016", "question": "What fusion method combines dense and sparse rankings?", "expected_answer_contains": "RRF", "expected_source": "retrieval", "should_be_sufficient": True},
    {"id": "e017", "question": "What is the reranker scoring based on?", "expected_answer_contains": "overlap", "expected_source": "retrieval", "should_be_sufficient": True},
    # --- answerable: embeddings ---
    {"id": "e018", "question": "What happens when no API key is set?", "expected_answer_contains": "fallback", "expected_source": "embeddings", "should_be_sufficient": True},
    {"id": "e019", "question": "Which embedding model is the default?", "expected_answer_contains": "text-embedding-3-small", "expected_source": "embeddings", "should_be_sufficient": True},
    # --- answerable: citations / sufficiency ---
    {"id": "e020", "question": "What is returned when evidence is insufficient?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
    {"id": "e021", "question": "How is an answer grounded?", "expected_answer_contains": "source", "expected_source": "citations", "should_be_sufficient": True},
    # --- answerable: deployment / stack ---
    {"id": "e022", "question": "How is the FastAPI service started?", "expected_answer_contains": "uvicorn", "expected_source": "deployment", "should_be_sufficient": True},
    {"id": "e023", "question": "What runs in CI on every push?", "expected_answer_contains": "pytest", "expected_source": "deployment", "should_be_sufficient": True},
    {"id": "e024", "question": "How is the service containerized?", "expected_answer_contains": "Docker", "expected_source": "deployment", "should_be_sufficient": True},
    # --- unanswerable: must return insufficient_evidence ---
    {"id": "e025", "question": "Who is the president of Mars?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
    {"id": "e026", "question": "What is the GDP of Wakanda?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
    {"id": "e027", "question": "What is the airspeed velocity of an unladen swallow?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
    {"id": "e028", "question": "Who wrote the screenplay for the 1996 film The Rock?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
    {"id": "e029", "question": "What is the capital of Narnia?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
    {"id": "e030", "question": "How do you bake a sourdough loaf?", "expected_answer_contains": "insufficient_evidence", "expected_source": "", "should_be_sufficient": False},
]
