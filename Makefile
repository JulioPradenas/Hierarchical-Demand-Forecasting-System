.PHONY: test lint typecheck check format

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

typecheck:
	uv run mypy src

check: lint typecheck test

format:
	uv run ruff format src tests
