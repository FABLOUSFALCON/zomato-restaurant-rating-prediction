#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = newAge
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python
CONDA_ENV_FILE = environment.yml

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies from newAge environment (RECOMMENDED)
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file $(CONDA_ENV_FILE) --prune

## Install conda dependencies from complete conda list
.PHONY: requirements-conda
requirements-conda:
	conda install --name $(PROJECT_NAME) --file requirements_conda.txt

## Install pip dependencies (fallback - incomplete package list)
.PHONY: requirements-pip
requirements-pip:
	pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Type checking with mypy
.PHONY: type-check
type-check:
	mypy zomato_prediction/

## Run tests
.PHONY: test
test:
	python -m pytest tests -v

## Run tests with coverage
.PHONY: test-cov
test-cov:
	python -m pytest tests --cov=zomato_prediction --cov-report=html --cov-report=term

## Set up newAge environment (RECOMMENDED - battle-tested environment)
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f $(CONDA_ENV_FILE)
	@echo ">>> newAge conda env created. Activate with:\nconda activate $(PROJECT_NAME)"

## Start the FastAPI development server
.PHONY: serve
serve:
	uvicorn zomato_prediction.api:app --reload --host 0.0.0.0 --port 8000

## Start MLflow tracking server
.PHONY: mlflow
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

## Start Redis server (if installed locally)
.PHONY: redis
redis:
	redis-server

## Build Docker image
.PHONY: docker-build
docker-build:
	docker build -t zomato-prediction:latest .

## Run Docker container
.PHONY: docker-run
docker-run:
	docker run -p 8000:8000 zomato-prediction:latest

## Start full stack with docker-compose
.PHONY: docker-up
docker-up:
	docker-compose up -d

## Stop docker-compose stack
.PHONY: docker-down
docker-down:
	docker-compose down

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) zomato_prediction/dataset.py

## Train models
.PHONY: train
train: requirements
	$(PYTHON_INTERPRETER) -m zomato_prediction.modeling.train

## Generate predictions
.PHONY: predict
predict: requirements
	$(PYTHON_INTERPRETER) -m zomato_prediction.modeling.predict


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
