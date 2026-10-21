COMPOSE ?= docker-compose
API_URL ?= http://localhost:8000
APP_TOKEN ?= demo-app-token

.PHONY: install lint format typecheck test up down logs token sample-pdf

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy app tests

test:
	uv run pytest

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api worker

token:
	@curl -s -X POST $(API_URL)/auth/token \
		-H 'Content-Type: application/json' \
		-d '{"app_token": "$(APP_TOKEN)"}'
	@echo

sample-pdf:
	uv run python examples/make_sample_pdf.py
