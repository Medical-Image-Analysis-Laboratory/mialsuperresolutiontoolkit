#!/bin/sh
SCRIPTSDIR=$(cd "$(dirname "$0")"; pwd)
BASEDIR="$(dirname "$SCRIPTSDIR")"
cd "$BASEDIR"

echo "Building documentation in $BASEDIR/documentation/_build/html"

cd "$BASEDIR/documentation"
make clean
make html
