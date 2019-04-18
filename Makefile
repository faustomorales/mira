# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = ./docs
BUILDDIR      = ./docs/_build
WORKDIR = /usr/src/mira
NOTEBOOK_PORT = 5000
DOCUMENTATION_PORT = 5001
IMAGE_NAME = mira
VOLUMES = -v $(PWD):/usr/src/mira
JUPYTER_OPTIONS := --ip=0.0.0.0 --port=$(NOTEBOOK_PORT) --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .
lab-server:
	docker run -it $(VOLUMES) -p $(NOTEBOOK_PORT):$(NOTEBOOK_PORT) -w $(WORKDIR) $(IMAGE_NAME) jupyter lab $(JUPYTER_OPTIONS)
documentation-server:
	docker run -it $(VOLUMES) -p $(DOCUMENTATION_PORT):8000 -w $(WORKDIR)  $(IMAGE_NAME) sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) --host 0.0.0.0 $(O)
.PHONY: tests
tests:
	docker run -it $(VOLUMES) -w $(WORKDIR) $(IMAGE_NAME) pytest
bash:
	docker run -it $(VOLUMES) -w $(WORKDIR) $(IMAGE_NAME) bash