.PHONY: dev test eval lint install clean

install:
	pip install -r requirements.txt

dev:
	uvicorn src.rag:app --reload --port 8000 2>/dev/null || python -m src.rag serve

test:
	pytest tests/ -v

eval:
	python -m src.rag eval

lint:
	python -m py_compile src/rag.py tests/test_rag.py

ci: lint test eval

clean:
	rm -rf index/*.json __pycache__/ .pytest_cache/
