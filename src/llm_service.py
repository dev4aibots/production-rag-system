"""Server-side LLM generation with rate limit and spend cap.

Exposes POST /generate — used by the production pipeline when the LLM
path is enabled. Protections:

  - rate limit: 10 requests per minute per client IP (sliding window)
  - spend cap:  US dollars per day; hard-stops above the cap
  - key is read from env only (OPENAI_API_KEY), never exposed
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

app = FastAPI(title="Production RAG System — LLM service", version="1.0.0")

RATE_LIMIT_PER_MIN = int(os.environ.get("LLM_RATE_LIMIT", "10"))
DAILY_SPEND_CAP_USD = float(os.environ.get("LLM_DAILY_SPEND_CAP", "1.00"))
MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
COST_INPUT_PER_1M = 0.15
COST_OUTPUT_PER_1M = 0.60

_rate_windows: dict[str, deque] = defaultdict(deque)
_spend_log: list[tuple[float, float]] = []
_SPEND_FILE = Path(__file__).resolve().parent.parent / "evals" / "spend-log.jsonl"


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _load_spend_today() -> float:
    today = datetime.now(timezone.utc).date().isoformat()
    total = 0.0
    if not _SPEND_FILE.exists():
        return 0.0
    try:
        lines = _SPEND_FILE.read_text(encoding="utf-8").splitlines()
        for line in lines:
            rec = json.loads(line)
            if rec.get("date") == today:
                total += rec.get("cost_usd", 0.0)
    except Exception:
        pass
    return total


def _log_spend(cost_usd: float) -> None:
    rec = {"date": datetime.now(timezone.utc).date().isoformat(), "cost_usd": round(cost_usd, 8), "ts": time.time()}
    _SPEND_FILE.parent.mkdir(exist_ok=True)
    with open(_SPEND_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


class GenerateRequest(BaseModel):
    context: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class GenerateResponse(BaseModel):
    answer: str
    model: str
    cost_usd: float
    latency_ms: int


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = _client_ip(request)
    now = time.time()
    window = _rate_windows[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded (10 req/min)"})
    window.append(now)
    return await call_next(request)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, x_api_key: Optional[str] = Header(None)):
    # Rate limiting is applied by the middleware (per client IP).
    expected_key = os.environ.get("LLM_SERVICE_KEY", "")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if _load_spend_today() >= DAILY_SPEND_CAP_USD:
        raise HTTPException(status_code=429, detail="Daily spend cap reached")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    start = time.time()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        system = (
            "You answer using ONLY the provided context. "
            "Cite sources inline as [Source: name]. "
            "If the answer is not supported by the context, reply exactly: insufficient_evidence. "
            "Do not invent facts."
        )
        user = f"Context:\n{req.context}\n\nQuestion: {req.question}"
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        answer_text = resp.choices[0].message.content or "insufficient_evidence"
        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = (input_tokens / 1e6) * COST_INPUT_PER_1M + (output_tokens / 1e6) * COST_OUTPUT_PER_1M
        _log_spend(cost)
        latency_ms = int((time.time() - start) * 1000)
        return GenerateResponse(answer=answer_text, model=MODEL, cost_usd=round(cost, 8), latency_ms=latency_ms)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM upstream error: {exc}")


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "model": MODEL, "spend_today_usd": round(_load_spend_today(), 8)}
