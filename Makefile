.PHONY: install format lint typecheck test check run infra-up infra-down

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

check: lint typecheck test

run:
	$(BIN)/uvicorn apps.api.main:app --reload

infra-up:
	docker compose up -d --wait

infra-down:
	docker compose down
