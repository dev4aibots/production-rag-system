FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn==0.32.0 \
    pydantic==2.9.2 \
    openai==1.65.0

COPY src/ ./src/
COPY data/ ./data/
COPY Makefile ./

EXPOSE 8000

CMD ["uvicorn", "src.rag:app", "--host", "0.0.0.0", "--port", "8000"]
