.PHONY: install install-all lint format test test-all tasks report clean docker-build

CONFIG ?= configs/default.yaml

install:
	pip install -e ".[dev]"

install-all:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,all]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

test:
	pytest -q -m "not slow"

test-all:
	pytest -q

tasks:
	loyalty-spend tasks --config $(CONFIG)

report:
	loyalty-spend report --config $(CONFIG)

docker-build:
	docker build -t loyalty-spend:latest .

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
