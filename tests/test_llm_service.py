"""Verify LLM service protections (rate limit, spend cap, auth) — no real OpenAI key needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.pop("OPENAI_API_KEY", None)
os.environ["LLM_SERVICE_KEY"] = "test-key"

from fastapi.testclient import TestClient  # noqa: E402
from llm_service import app  # noqa: E402

client = TestClient(app)
HEADERS = {"X-API-Key": "test-key"}


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_generate_without_api_key_returns_503():
    r = client.post("/generate", json={"context": "c", "question": "q"}, headers=HEADERS)
    assert r.status_code == 503


def test_generate_without_service_key_returns_401():
    os.environ["OPENAI_API_KEY"] = "sk-dummy"
    r = client.post("/generate", json={"context": "c", "question": "q"})
    assert r.status_code == 401


def test_rate_limit_triggers_429():
    codes = []
    for _ in range(14):
        r = client.post("/generate", json={"context": "c", "question": "q"}, headers=HEADERS)
        codes.append(r.status_code)
        if r.status_code == 429:
            break
    assert codes[-1] == 429
