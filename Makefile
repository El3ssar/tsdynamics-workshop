
# TSDynamics workshop — convenience targets. Requires Python >= 3.12.
.PHONY: help setup lab data execute clean

help:
	@echo "make setup    - create .venv and install everything (uv if available, else pip)"
	@echo "make lab      - launch JupyterLab"
	@echo "make data     - regenerate the capstone datasets"
	@echo "make execute  - run every notebook end-to-end (validates them)"
	@echo "make clean    - remove caches and checkpoints"

setup:
	./setup.sh

lab:
	./.venv/bin/jupyter lab

data:
	cd capstone && ../.venv/bin/python generate_data.py

execute:
	@for nb in notebooks/*.ipynb; do \
		echo ">> executing $$nb"; \
		./.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=900 "$$nb" || exit 1; \
	done

clean:
	find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
