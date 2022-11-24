SHELL = /bin/bash

# Environment
.PHONY: setup
setup:
	python3 -m venv ~/ccfd && \
	source ~/ccfd/bin/activate && \
	pip3 install -r requirements.txt && \
	cp env.toml.example env.toml

# Cleaning
.PHONY: clean
clean: 
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	find . | grep -E ".trash" | xargs rm -rf
	rm -f .coverage
	black .

# Migrate data to db
.PHONY: migrate
migrate: 
	python3 src/prepare.py
	python3 src/migration.py

# run pipe
.PHONY: run
run: 
	python3 src/prepare.py
	python3 src/featurize.py
	python3 src/train.py
	make clean

.PHONY: help
help:
	@echo "Commands:"
	@echo "setup   : creates a virtual environment (ccfd) for the project."
	@echo "clean   : deletes all unnecessary files and executes style formatting."
	@echo "migrate : prepare and migrate data to database."