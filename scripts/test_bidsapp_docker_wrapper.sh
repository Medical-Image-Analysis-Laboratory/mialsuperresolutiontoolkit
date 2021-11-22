#!/bin/sh
SCRIPTSDIR=$(cd "$(dirname "$0")"; pwd)
BASEDIR="$(dirname "$SCRIPTSDIR")"

mialsuperresolutiontoolkit_docker \
    "$BASEDIR/data" \
    "$BASEDIR/data/derivatives" \
    participant --participant_label 01 \
    --param_file "$BASEDIR/data/code/participants_params.json" \
    --nipype_nb_of_cores 1 \
    --openmp_nb_of_cores 1
