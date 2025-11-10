.PHONY: install install-all lint typecheck format test test-all mutation docs tasks report clean docker-build docker-serve

CONFIG ?= configs/default.yaml

install:
	pip install -e ".[dev]"

install-all:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev,all]"

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

format:
	ruff check . --fix
	ruff format .

test:
	pytest -q -m "not slow"

test-all:
	pytest -q

mutation:
	python scripts/mutation_check.py src/loyalty_spend/business/rewards.py \
		tests/test_rewards.py tests/test_properties_rewards.py \
		tests/test_memory_bounded.py tests/test_rewards_mutants.py

docs:
	mkdocs build --strict

tasks:
	loyalty-spend tasks --config $(CONFIG)

report:
	loyalty-spend report --config $(CONFIG)

docker-build:
	docker build --target cli -t loyalty-spend:latest .

docker-serve:
	docker build --target serve -t loyalty-spend-api:latest .
	docker run --rm -p 8000:8000 -v $(PWD)/artifacts:/app/artifacts loyalty-spend-api:latest

clean:
	rm -rf artifacts .pytest_cache .ruff_cache build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
