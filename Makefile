.PHONY: install format lint typecheck test test-integration test-live-model benchmark-chunking benchmark-native-indexing build-evaluation-dataset seed-evaluation evaluate compare-frameworks smoke-ollama smoke-api check check-all run web-install web-dev web-build web-test web-check infra-up infra-down

PYTHON ?= python3.12
RAGLAB_FRAMEWORK ?= custom
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

benchmark-native-indexing:
	$(BIN)/python scripts/benchmark_native_indexing.py

build-evaluation-dataset:
	$(BIN)/python scripts/build_evaluation_dataset.py

seed-evaluation:
	$(BIN)/python scripts/seed_evaluation.py

evaluate:
	$(BIN)/python scripts/run_evaluation.py --model $(RAGLAB_LLM_MODEL) --framework $(RAGLAB_FRAMEWORK)

compare-frameworks:
	$(BIN)/python scripts/compare_frameworks.py --model $(RAGLAB_LLM_MODEL)

smoke-ollama:
	$(BIN)/python scripts/smoke_ollama.py --model $(RAGLAB_LLM_MODEL)

smoke-api:
	$(BIN)/python scripts/smoke_api.py --model $(RAGLAB_LLM_MODEL)

check: lint typecheck test

check-all: check web-check

run:
	$(BIN)/uvicorn apps.api.main:app --reload

web-install:
	npm --prefix apps/web ci

web-dev:
	npm --prefix apps/web run dev

web-build:
	npm --prefix apps/web run build

web-test:
	npm --prefix apps/web run test

web-check:
	npm --prefix apps/web run lint
	npm --prefix apps/web run typecheck
	npm --prefix apps/web run test
	npm --prefix apps/web run build

infra-up:
	docker compose up -d --wait

infra-down:
	docker compose down
