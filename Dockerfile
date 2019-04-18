FROM tensorflow/tensorflow:1.13.1-py3
RUN apt-get update && apt-get install -y \
	libsm6 \
	libxrender1 \
	curl \
	net-tools \
	git \
	libglib2.0-0

# Copy files and install Python dependencies
COPY . /usr/src/mira
RUN pip install -e /usr/src/mira[tests,docs]
RUN pip install jupyterlab