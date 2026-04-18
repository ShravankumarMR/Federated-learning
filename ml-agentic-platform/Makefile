.PHONY: install lint type-check test run

install:
	pip install -e .[dev]

lint:
	ruff check src tests

type-check:
	mypy src

test:
	pytest

run:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
