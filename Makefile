.PHONY: install lint format test test-all eval eval-offline up down logs migrate run

install:  ## Create .venv and install dev dependencies
	python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pre-commit install

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check .

format:
	.venv/bin/ruff check --fix . && .venv/bin/ruff format .

test:  ## Fast suite (no ChromaDB model download, no network)
	.venv/bin/pytest -m "not slow and not network" --cov

test-all:
	.venv/bin/pytest -m "not network" --cov

eval:  ## Run the eval suite against the real API (spends credit; capped by --max-cost)
	.venv/bin/python -m evals.run

eval-offline:  ## Eval suite without internet-dependent tasks
	.venv/bin/python -m evals.run --offline

run:  ## Run the API locally (needs Redis; Postgres optional)
	.venv/bin/uvicorn api.main:app --reload

migrate:
	.venv/bin/alembic upgrade head

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api
