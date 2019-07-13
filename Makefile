# You can set these variables from the command line.
WORKDIR = /usr/src
NOTEBOOK_PORT = 5000
DOCUMENTATION_PORT = 5001
DOCKER_ARGS = $(VOLUMES) -w $(WORKDIR) --rm 
IMAGE_NAME = mira
VOLUME_NAME = $(IMAGE_NAME)_venv
VOLUMES = -v $(PWD):/usr/src -v $(VOLUME_NAME):/usr/src/.venv --rm
JUPYTER_OPTIONS := --ip=0.0.0.0 --port=$(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

.PHONY: build
build:
	docker build --rm --force-rm -t $(IMAGE_NAME) .
	@-docker volume rm $(VOLUME_NAME)
init:
	PIPENV_VENV_IN_PROJECT=true pipenv install --dev --skip-lock
lab-server:
	docker run -it $(DOCKER_ARGS) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) $(IMAGE_NAME) pipenv run jupyter lab $(JUPYTER_OPTIONS)
documentation-server:
	docker run -it $(DOCKER_ARGS)-p $(DOCUMENTATION_PORT):$(DOCUMENTATION_PORT) $(IMAGE_NAME) pipenv run sphinx-autobuild -b html "docs" "docs/_build/html" --host 0.0.0.0 --port $(DOCUMENTATION_PORT) $(O)
.PHONY: tests
test:
	docker run -it $(DOCKER_ARGS) $(IMAGE_NAME) pipenv run pytest
.PHONY: docs
docs:
	docker run -it $(DOCKER_ARGS) $(IMAGE_NAME) pipenv run sphinx-build -b html "docs" "docs/dist"
bash:
	docker run -it $(DOCKER_ARGS) $(IMAGE_NAME) bash