# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = ./docs
BUILDDIR      = ./docs/_build
WORKDIR = /usr/src
NOTEBOOK_PORT = 5000
DOCUMENTATION_PORT = 5001
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
build:
	docker build -t $(IMAGE_NAME) .
lab-server:
	docker run -it $(VOLUMES) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) -w $(WORKDIR) $(IMAGE_NAME) pipenv run jupyter lab $(JUPYTER_OPTIONS)
documentation-server:
	docker run -it $(VOLUMES) -p $(DOCUMENTATION_PORT):8000 -w $(WORKDIR)  $(IMAGE_NAME) sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --host 0.0.0.0 $(O)
.PHONY: tests
test:
	docker run -it $(VOLUMES) -w $(WORKDIR) $(IMAGE_NAME) pipenv run pytest
bash:
	docker run -it $(VOLUMES) -w $(WORKDIR) $(IMAGE_NAME) bash