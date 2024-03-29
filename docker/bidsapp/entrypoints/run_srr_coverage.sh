#!/bin/bash
echo "User: $(id -un "$USER")" && echo "Group: $(id -gn "$USER")" && \
export && \
echo "SHELL: $SHELL" && \
echo "PATH: $PATH" && \
xvfb-run -a coverage run --source=pymialsrtk \
/app/run.py "$@" \
|& tee /bids_dir/code/log.txt && \
coverage html -d /bids_dir/code/coverage_html && \
coverage xml -o /bids_dir/code/coverage.xml && \
coverage json -o /bids_dir/code/coverage.json
