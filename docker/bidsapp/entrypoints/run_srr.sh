#!/bin/bash
echo "User: $(id -un "$USER")" && echo "Group: $(id -gn "$USER")" && \
conda activate "${MY_CONDA_PY3ENV}" && \
xvfb-run -a python /opt/mialsuperresolutiontoolkit/docker/bidsapp/run.py "$@"