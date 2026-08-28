.PHONY: sync build binary check-version lint test

sync:
	python3 -m venv .venv
	.venv/bin/python -m pip install --requirement requirements-dev.lock
	.venv/bin/python -m pip install --no-build-isolation --no-deps --editable .

check-version:
	.venv/bin/python scripts/check_versions.py

build: check-version
	.venv/bin/python -m build

binary: check-version
	.venv/bin/pyinstaller --clean --noconfirm betterborg.spec

lint:
	.venv/bin/ruff check .
	npm --prefix npm run lint

test:
	.venv/bin/pytest
	npm --prefix npm test
