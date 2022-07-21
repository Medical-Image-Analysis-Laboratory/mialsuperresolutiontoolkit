#!/bin/bash
echo "User: $(id -un "$USER")" && echo "Group: $(id -gn "$USER")" && \
export && \
echo "SHELL: $SHELL" && \
echo "PATH: $PATH" && \
. activate "${MY_CONDA_PY3ENV}" && \
xvfb-run -a python /opt/mialsuperresolutiontoolkit/pymialsrtk/run_preprocess.py "$@"