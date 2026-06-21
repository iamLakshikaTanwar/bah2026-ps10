# irchroma — developer Makefile (BAH 2026 PS-10)
# Usage: `make <target>`. See `make help` for the list.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
SRC    := src
PKG    := irchroma
CONFIG ?= configs/default.yaml
# Ensure the src/ layout is importable for ad-hoc invocations.
export PYTHONPATH := $(SRC):$(PYTHONPATH)

.DEFAULT_GOAL := help
.PHONY: help install install-dev install-all demo train eval test lint format \
        typecheck serve smoke config-dump clean

help:  ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install core runtime dependencies (editable package).
	$(PIP) install -e .

install-dev:  ## Install core + dev tooling (pytest/ruff/black/mypy).
	$(PIP) install -e ".[dev]"

install-all:  ## Install everything (gee, serve, trt, detect, energy, dev).
	$(PIP) install -e ".[gee,serve,trt,detect,energy,dev]"

smoke:  ## Import-only smoke check for the config + interfaces contract (no torch needed).
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(SRC)'); \
import irchroma.config as c, irchroma.interfaces as i; \
print('OK contract import; TORCH_AVAILABLE =', i.TORCH_AVAILABLE); \
print('Config sections:', [x for x in dir(c) if 'Config' in x])"

config-dump:  ## Regenerate configs/default.yaml from the dataclass defaults.
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(SRC)'); \
from irchroma.config import Config; Config().to_yaml('$(CONFIG)'); \
print('Wrote $(CONFIG)')"

demo:  ## Run the synthetic-data end-to-end demo (no network/credentials/GPU).
	$(PYTHON) -m $(PKG).train.cli --demo --config $(CONFIG)

train:  ## Train (override CONFIG=... ; e.g. make train CONFIG=configs/landsat.yaml).
	$(PYTHON) -m $(PKG).train.cli --config $(CONFIG)

eval:  ## Run the 6-family evaluation suite on a checkpoint/results dir.
	$(PYTHON) -m $(PKG).metrics.cli --config $(CONFIG)

serve:  ## Launch the FastAPI/TiTiler tile server (requires the [serve] extra).
	$(PYTHON) -m $(PKG).serve.app --config $(CONFIG)

test:  ## Run the test suite.
	$(PYTHON) -m pytest

lint:  ## Lint with ruff.
	$(PYTHON) -m ruff check $(SRC)

format:  ## Auto-format with black + ruff --fix.
	$(PYTHON) -m black $(SRC)
	$(PYTHON) -m ruff check --fix $(SRC)

typecheck:  ## Static type-check with mypy.
	$(PYTHON) -m mypy $(SRC)/$(PKG)

clean:  ## Remove caches and build artifacts.
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
