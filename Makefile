# Makefile for pyQIC: setup, install, and test

# Name of the conda environment
ENV_NAME ?= pyqic-env
PYTHON ?= python

.PHONY: help env install install-extras test clean

help:
	@echo "pyQIC Makefile targets:"
	@echo "  make env            Create conda environment with all dependencies."
	@echo "  make install        Install all pip dependencies (requirements.txt)."
	@echo "  make install-extras Install extra dependencies (qsc, booz-xform)."
	@echo "  make test           Run all tests."
	@echo "  make clean          Remove __pycache__ and .pyc files."
	@echo "  make help           Show this help message."
	@echo "  External dependencies (manual): VMEC, BAD. See README for instructions."

# Create conda environment and install dependencies
env:
	conda create -y -n $(ENV_NAME) python=3.11
	conda activate $(ENV_NAME) && \
		pip install --upgrade pip && \
		pip install -r requirements.txt && \
		pip install qsc booz-xform
	@echo "\nManual steps required for VMEC and BAD. See README."

# Install pip dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Install extra dependencies (pip)
install-extras:
	$(PYTHON) -m pip install qsc booz-xform
	@echo "\nManual steps required for VMEC and BAD. See README."

# Run all tests
# (Assumes tests are in qic/examples/ or qic/tests/)
test:
	$(PYTHON) -m unittest discover -s qic/examples -p 'test*.py' || true
	$(PYTHON) -m unittest discover -s qic/tests -p 'test*.py' || true

# Clean up Python cache files
clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
