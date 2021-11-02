#!/bin/sh
SCRIPTSDIR=$(cd "$(dirname "$0")"; pwd)
BASEDIR="$(dirname "$SCRIPTSDIR")"

# DOCKER_IMAGE="docker.io/sebastientourbier/mialsuperresolutiontoolkit-bidsapp:v2.0.3"
DOCKER_IMAGE="sebastientourbier/mialsuperresolutiontoolkit-bidsapp:v2.0.3"

docker run -it --rm -u $(id -u):$(id -g) \
    -v "$BASEDIR/data":/bids_dir \
    -v "$BASEDIR/data/derivatives":/output_dir \
    "$DOCKER_IMAGE" \
    /bids_dir /output_dir participant --participant_label 01 \
    --param_file /bids_dir/code/participants_params.json \
    --nipype_nb_of_cores 2 \
    --openmp_nb_of_cores 1
