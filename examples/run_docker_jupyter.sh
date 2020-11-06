#!/usr/bin/sh

mialsrtk_dir="/home/hkebiri/mialsuperresolutiontoolkit"
data_dir="/home/hkebiri/mialsuperresolutiontoolkit/data"

port=8888

version=v2.0.0-beta-20190906

cmd="docker run --rm"
cmd="$cmd -v "${mialsrtk_dir}/notebooks":/app/notebooks"
cmd="$cmd -v "${mialsrtk_dir}":/app/mialsuperresolutiontoolkit"
cmd="$cmd -v "${data_dir}":/fetaldata"
cmd="$cmd -p ${port}:${port}"
cmd="$cmd -t sebastientourbier/mialsuperresolutiontoolkit-jupyter:${version}"

eval $cmd

