.PHONY: install lint format test typecheck qa

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff format

test:
	pytest -q -o addopts=''

typecheck:
	mypy --ignore-missing-imports .

qa: lint typecheck test
