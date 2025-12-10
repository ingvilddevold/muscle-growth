FROM ghcr.io/fenics/dolfinx/lab:v0.10.0

COPY . /repo
WORKDIR /repo

RUN python3 -m pip install ".[docs]"

ENTRYPOINT ["/bin/bash"]