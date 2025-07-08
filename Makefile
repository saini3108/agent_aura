# Define the Python interpreter
PYTHON ?= python

# Define where requirements live (change if you use a requirements/ folder)
REQ_LOCAL = requirements/local.txt
REQ_PROD = requirements/production.txt

# Paths
SRC_DIR = agent_service
TEST_DIR = tests

# ===== INSTALLATION =====

.PHONY: install
install:
	pip install --upgrade pip
	pip install -r $(REQ_LOCAL)

.PHONY: install-prod
install-prod:
	pip install -r $(REQ_PROD)


# ===== DEVELOPMENT COMMANDS =====
.PHONY: run
run:
	$(PYTHON) -m $(SRC_DIR).main

.PHONY: dev
dev:
	uvicorn agent_service.main:app --reload


# ===== CODE QUALITY =====

.PHONY: lint
lint:
	ruff check . --fix

.PHONY: format
format:
	ruff format .
	black $(SRC_DIR)

.PHONY: typecheck
typecheck:
	mypy $(SRC_DIR)

.PHONY: precheck
precheck: lint format typecheck


# ===== TESTING =====

.PHONY: test
test:
	pytest $(TEST_DIR)

.PHONY: test-cov
test-cov:
	coverage run -m pytest $(TEST_DIR)
	coverage report -m


# ===== FLOWS =====

.PHONY: flow
flow:
	$(PYTHON) scripts/run_validation_flow.py


# ===== UTILITIES =====

.PHONY: clean-pyc
clean-pyc:
	find . -name "*.pyc" -exec rm -f {} \;

.PHONY: clean
clean: clean-pyc

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make install           Install dev dependencies"
	@echo "  make install-prod      Install prod dependencies"
	@echo "  make dev               Run FastAPI with live reload"
	@echo "  make run               Run main.py"
	@echo "  make lint              Run ruff linter"
	@echo "  make format            Format code with ruff + black"
	@echo "  make typecheck         Run mypy"
	@echo "  make precheck          Run lint + format + typecheck"
	@echo "  make test              Run pytest"
	@echo "  make test-cov          Run tests with coverage"
	@echo "  make flow              Run sample LangGraph flow"
	@echo "  make clean             Remove .pyc files"
