# Credit Default MLOps -- run `make help` for the full list.
.DEFAULT_GOAL := help
SHELL := /bin/bash
UV := uv run
COMPOSE := docker compose
TF := terraform -chdir=infra/gcp

.PHONY: help setup lint format typecheck test test-cov check \
        ingest split train evaluate promote pipeline \
        serve drift simulate-drift \
        up down logs ps clean \
        docker-build cloud-init cloud-plan cloud-up cloud-down cloud-cost

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- dev ------

setup: ## Install all dependencies and pre-commit hooks
	uv sync --all-extras
	$(UV) pre-commit install

lint: ## Lint with ruff
	$(UV) ruff check src tests scripts flows

format: ## Auto-format and fix lint
	$(UV) ruff format src tests scripts flows
	$(UV) ruff check --fix src tests scripts flows

typecheck: ## Static type check with mypy
	$(UV) mypy

test: ## Run the test suite
	$(UV) pytest

test-cov: ## Run tests with a coverage report
	$(UV) pytest --cov --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Everything CI runs

# ------------------------------------------------------------- pipeline ----

ingest: ## Download the UCI dataset
	$(UV) python -m credit_default.data.ingest

split: ## Rebuild the cohort splits (also restores a clean cohort after drift simulation)
	$(UV) python -m credit_default.data.split

train: ## Train candidates, log to MLflow, register the best as @challenger
	$(UV) python -m credit_default.train

evaluate: ## Run the model quality gate (exits non-zero on failure)
	$(UV) python -m credit_default.evaluate

promote: ## Promote @challenger to @champion, if the gate passes
	$(UV) python -m credit_default.promote

pipeline: ## Run the full DVC pipeline (only re-runs what changed)
	$(UV) dvc repro

# ------------------------------------------------------------ monitoring ---

drift: ## Compare the current cohort against reference
	$(UV) python -m credit_default.monitoring.drift

simulate-drift: ## Inject SYNTHETIC drift to demo the monitor (undo with: make split)
	$(UV) python scripts/simulate_drift.py

# --------------------------------------------------------------- serving ---

serve: ## Run the API locally against the on-disk model
	MODEL_SOURCE=local $(UV) uvicorn credit_default.api.main:app --reload --port 8000

# ----------------------------------------------------------------- stack ---

up: ## Start the full local stack (API, MLflow, Postgres, Prometheus, Grafana)
	$(COMPOSE) up -d --build
	@echo
	@echo "  API         http://localhost:8000/docs"
	@echo "  MLflow      http://localhost:5001"
	@echo "  Prometheus  http://localhost:9090"
	@echo "  Grafana     http://localhost:3000  (admin / admin)"

down: ## Stop the stack
	$(COMPOSE) down

logs: ## Follow stack logs
	$(COMPOSE) logs -f

ps: ## Show stack status
	$(COMPOSE) ps

docker-build: ## Build the API image
	docker build -f docker/api.Dockerfile -t credit-default-api:local .

clean: ## Remove generated artefacts (keeps raw data)
	rm -rf reports .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

# ----------------------------------------------------------------- cloud ---
# Everything below targets GCP's always-free tier. `cloud-down` is not optional
# housekeeping -- run it when you finish a demo.

cloud-init: ## Initialise Terraform
	$(TF) init

cloud-plan: ## Preview cloud changes
	$(TF) plan

cloud-up: ## Deploy to GCP (budget alert is created first)
	$(TF) apply -auto-approve
	@echo
	@echo "Remember: run 'make cloud-down' when you are finished."

cloud-down: ## TEAR DOWN all cloud resources
	$(TF) destroy -auto-approve

cloud-cost: ## Show month-to-date spend
	@gcloud billing accounts list
	@echo "Full detail: https://console.cloud.google.com/billing"
