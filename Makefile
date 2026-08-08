# Convenience wrappers. Everything here is a thin alias for a command already
# documented in the README -- the Makefile exists so a new developer does not
# have to get the invocation exactly right on their first day.

.PHONY: help setup test lint repro pipeline clean mlflow push pull metrics dag

help:
	@echo "make setup    - create .venv and install pinned dependencies"
	@echo "make test     - run the pytest suite with coverage"
	@echo "make lint     - run flake8"
	@echo "make repro    - run the DVC pipeline (only stale stages)"
	@echo "make pipeline - force a full re-run of every stage"
	@echo "make metrics  - show the metrics from the last run"
	@echo "make dag      - print the pipeline DAG"
	@echo "make mlflow   - launch the MLflow UI at http://localhost:5000"
	@echo "make push     - upload data and models to the DVC remote"
	@echo "make pull     - download data and models from the DVC remote"
	@echo "make clean    - delete generated artifacts (keeps the DVC cache)"

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "Done. Activate with: source .venv/bin/activate"

test:
	.venv/bin/pytest --cov=src --cov-report=term-missing

lint:
	.venv/bin/flake8 src tests --count --statistics

repro:
	PATH="$(PWD)/.venv/bin:$$PATH" .venv/bin/dvc repro

pipeline:
	PATH="$(PWD)/.venv/bin:$$PATH" .venv/bin/dvc repro --force

metrics:
	.venv/bin/dvc metrics show

dag:
	.venv/bin/dvc dag

mlflow:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

push:
	.venv/bin/dvc push

pull:
	.venv/bin/dvc pull

clean:
	rm -rf data/processed/* models reports/plots reports/metrics.json \
	       reports/model_comparison.csv .pytest_cache .coverage coverage.xml
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	@echo "Cleaned. Raw data and the DVC cache were left alone."
