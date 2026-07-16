.PHONY: install format lint typecheck test test-integration test-live-model benchmark-chunking smoke-ollama smoke-api check run infra-up infra-down

PYTHON ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e '.[dev]'

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

lint:
	$(BIN)/ruff format --check .
	$(BIN)/ruff check .

typecheck:
	$(BIN)/mypy

test:
	$(BIN)/pytest --cov --cov-report=term-missing

test-integration:
	$(BIN)/pytest -m integration

test-live-model:
	$(BIN)/pytest -m live_model

benchmark-chunking:
	$(BIN)/python scripts/benchmark_chunking.py

smoke-ollama:
	$(BIN)/python scripts/smoke_ollama.py --model $(RAGLAB_LLM_MODEL)

smoke-api:
	$(BIN)/python scripts/smoke_api.py --model $(RAGLAB_LLM_MODEL)

check: lint typecheck test

run:
	$(BIN)/uvicorn apps.api.main:app --reload

infra-up:
	docker compose up -d --wait

infra-down:
	docker compose down
