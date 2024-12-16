ENV_TYPE ?= venv
PYTHON_VERSION ?= 3.10
VENV_DIR := .venv
CONDA_ENV ?= onemod

.PHONY: list-vars check-python setup clean mypy pre-commit

list-vars:  ## List variables
	@echo "ENV_TYPE        = $(ENV_TYPE)"
	@echo "PYTHON_VERSION  = $(PYTHON_VERSION)"
	@echo "VENV_DIR        = $(VENV_DIR)"
	@echo "CONDA_ENV       = $(CONDA_ENV)"

check-python:  ## Check if the specified Python version is installed (venv only)
	@if [ "$(ENV_TYPE)" = "venv" ]; then \
		if ! command -v python$(PYTHON_VERSION) >/dev/null 2>&1; then \
			echo "Error: Python $(PYTHON_VERSION) is not installed."; \
			exit 1; \
		fi; \
	fi

setup: check-python  ## Set up the development environment (venv or conda)
ifeq ($(ENV_TYPE),venv)
	python$(PYTHON_VERSION) -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -U pip
	$(VENV_DIR)/bin/pip install -e ".[dev]"
	$(VENV_DIR)/bin/pre-commit install
else ifeq ($(ENV_TYPE),conda)
	conda create -n $(CONDA_ENV) python=$(PYTHON_VERSION) -y
	conda run -n $(CONDA_ENV) pip install -U pip
	conda run -n $(CONDA_ENV) pip install -e ".[dev]"
	conda run -n $(CONDA_ENV) pre-commit install
else
	$(error ENV_TYPE must be 'venv' or 'conda')
endif

mypy:  ## Run mypy
ifeq ($(ENV_TYPE),venv)
	$(VENV_DIR)/bin/mypy src/ tests/
else ifeq ($(ENV_TYPE),conda)
	conda run -n $(CONDA_ENV) mypy src/ tests/
endif

pre-commit:  ## Run pre-commit hooks
ifeq ($(ENV_TYPE),venv)
	$(VENV_DIR)/bin/pre-commit run --all-files
else ifeq ($(ENV_TYPE),conda)
	conda run -n $(CONDA_ENV) pre-commit run --all-files
endif

clean:  ## Remove virtual environment or conda environment
ifeq ($(ENV_TYPE),venv)
	rm -rf $(VENV_DIR)
else ifeq ($(ENV_TYPE),conda)
	conda remove -n $(CONDA_ENV) --all -y
endif

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
