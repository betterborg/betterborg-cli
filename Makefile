.PHONY: sync build lint test

sync:
	python3 -m venv .venv
	.venv/bin/python -m pip install --requirement requirements-dev.lock
	.venv/bin/python -m pip install --no-build-isolation --no-deps --editable .

build:
	.venv/bin/python -m build

lint:
	.venv/bin/ruff check .

test:
	.venv/bin/pytest
