.PHONY: install dev api dashboard test lint docker-up clean

install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev]"

api:
	uvicorn neuraldoc.api.main:app --reload --host 0.0.0.0 --port 8000

dashboard:
	streamlit run dashboard/app.py --server.port 8501

test:
	pytest -q

lint:
	ruff check src tests dashboard || true

docker-up:
	docker compose up --build

clean:
	rm -rf data/indices/* data/uploads/* data/*.db htmlcov .coverage
