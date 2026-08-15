# bandwidth-zen developer targets. Frontend targets are no-ops until M4
# (frontend deferred — see docs/CORRECTIONS.md).

BACKEND := cd backend && uv run

.PHONY: venv dev test test-fast lint fmt types validate plots docker

venv:
	cd backend && uv venv --python 3.11 && uv sync --all-groups

dev:
	@echo "backend-only phase: use 'uv run bwz --help' (API server lands at M4)"

test:
	$(BACKEND) pytest -q

test-fast:
	$(BACKEND) pytest -q -m "not validation and not slow"

lint:
	$(BACKEND) ruff check .
	$(BACKEND) ruff format --check .
	$(BACKEND) mypy

fmt:
	$(BACKEND) ruff format .
	$(BACKEND) ruff check --fix .

types:
	@echo "no-op until M4: frontend types are generated from OpenAPI"

validate:
	$(BACKEND) pytest -q -m validation -s

# Regenerates docs/plots/. Every figure is computed by calling analyze(), so
# these are outputs of the engine, not illustrations of it.
plots:
	$(BACKEND) --group plots python scripts/plot_roofline.py --chip a100_80gb --model llama3_8b
	$(BACKEND) --group plots python scripts/plot_roofline.py --chip chip_a --weights int8 \
		--model gemma3_4b --tokens 512 \
		--gemm 512,4096,4096 --gemm 128,4096,4096 --gemm 1,4096,4096

docker:
	@echo "no-op until M4: docker compose lands with the API + frontend"
