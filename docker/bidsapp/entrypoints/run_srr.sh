#!/bin/bash
echo "User: $(id -un "$USER")" && echo "Group: $(id -gn "$USER")" && \
export && \
echo "SHELL: $SHELL" && \
echo "PATH: $PATH" && \
"${CONDA_ACTIVATE}" && \
xvfb-run -a python /opt/mialsuperresolutiontoolkit/docker/bidsapp/run.py "$@"