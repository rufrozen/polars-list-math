SHELL := /bin/bash

UV_CMD ?= uv
PYTHON_VERSION ?= 3.12
PACKAGE_FILE ?= dist/polars-list-math-source.tar.gz
UV_DEV_RUN ?= $(UV_CMD) run --no-sync --group dev

.PHONY: init install lock develop format lint test build check-dist publish package clean

init:
	$(UV_CMD) sync --python $(PYTHON_VERSION) --all-extras --group dev --no-install-project

install:
	$(UV_CMD) sync --all-extras --group dev --no-install-project

lock:
	$(UV_CMD) lock

develop:
	$(UV_DEV_RUN) maturin develop

format:
	$(UV_DEV_RUN) ruff format .

lint:
	$(UV_DEV_RUN) ruff check .
	$(UV_DEV_RUN) ruff format --check .
	$(UV_DEV_RUN) pyright


test:
	$(UV_DEV_RUN) pytest

build:
	rm -rf build dist
	$(UV_CMD) build

check-dist: build
	$(UV_CMD) run --group publish twine check dist/*

publish: check-dist
	$(UV_CMD) run --group publish twine upload dist/polars_list_math-*

package:
	mkdir -p $(dir $(PACKAGE_FILE))
	git ls-files --cached --others --exclude-standard -z \
		| while IFS= read -r -d '' file; do \
			test ! -e "$$file" || printf '%s\0' "$$file"; \
		done \
		| tar --null --files-from - -czf $(PACKAGE_FILE)

clean:
	cargo clean --manifest-path rust/Cargo.toml
	rm -rf target build dist *.egg-info .pytest_cache .ruff_cache
	rm -f polars_list_math/*.so polars_list_math/*.pyd polars_list_math/*.dll polars_list_math/*.dylib
