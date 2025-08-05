.PHONY: docs protos

PKG_NAME:=mira

# Select specific Python tests to run using pytest selectors
# e.g., make test TEST_SCOPE='-m "not_integration" tests/api/'
TEST_SCOPE?=tests/

# Prefix for running commands on the host vs in Docker (e.g., dev vs CI)
EXEC:=uv run
SPHINX_AUTO_EXTRA:=


help:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Tips"
	@echo "----"
	@echo '- Run `make shell` to activate the project virtualenv in your shell'
	@echo '  e.g., make test TEST_SCOPE="-m not_integration tests/api/"'

protos: ## Build protobufs.
	protoc --python_out=mira/core protos/scene.proto

patch-thirdparty: ## Patch thirdparty module import structure.
	git submodule update
	git submodule init
	find mira/thirdparty/albumentations -name '*.py' -exec sed -i'.bak' -e 's/from albumentations/from mira.thirdparty.albumentations.albumentations/g' {} +
	find mira/thirdparty/albumentations -name '*.py' -exec sed -i'.bak' -e 's/from .domain_adaptation import \*//g' {} +
	find mira/thirdparty/smp -name '*.py' -exec sed -i'.bak' -e 's/from segmentation_models_pytorch/from mira.thirdparty.smp.segmentation_models_pytorch/g' {} +
	find mira/thirdparty -name '*.py.bak' -exec rm {} +

init:  patch-thirdparty ## Initialize the development environment.
	uv sync --extra clip --extra segmentation --extra detectors

format-check: ## Make black check source formatting
	@$(EXEC) black --diff --check $(PKG_NAME) tests

format: ## Make black unabashedly format source code
	@$(EXEC) black --exclude mira/thirdparty $(PKG_NAME) tests

package: patch-thirdparty ## Make a local build of the Python package, source dist and wheel
	@rm -rf dist
	uv build

test: ## Make pytest run tests
	@$(EXEC) pytest -vxrs $(TEST_SCOPE)

type-check: ## Make mypy check types
	@$(EXEC) mypy $(PKG_NAME) tests

lint-check: ## Make pylint lint the package
	@$(EXEC) pylint --rcfile pyproject.toml --jobs 0 $(PKG_NAME)

lab: ## Start a jupyter lab instance
	@$(EXEC) jupyter lab

check: format-check type-check lint-check test ## Run all CI/CD checks

docs: ## Make a local HTML doc server that updates on changes to from Sphinx source
	@$(EXEC) sphinx-autobuild -b html docs docs/build/html $(SPHINX_AUTO_EXTRA)