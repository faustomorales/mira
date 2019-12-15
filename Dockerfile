FROM python:3.7
RUN apt-get update && apt-get install -y \
	libsm6 \
	libxrender1 \
	curl \
	net-tools \
	git \
	libglib2.0-0

RUN pip install pipenv

# Install project
WORKDIR /usr/src
COPY ./setup* ./
COPY ./Pipfile* ./
COPY ./versioneer* ./
COPY ./Makefile Makefile
COPY ./mira/utils ./mira/utils
RUN make init