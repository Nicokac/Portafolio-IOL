.PHONY: install lint format lint-tests-fix test typecheck qa

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff format .

lint-tests-fix:
	ruff check tests --select I,F401 --fix
	ruff format tests

test:
	pytest -q -o addopts=''

typecheck:
	mypy --ignore-missing-imports .

qa: lint typecheck test

.PHONY: test_fast
test_fast:
	pytest -q -m "not slow and not live_yahoo"
