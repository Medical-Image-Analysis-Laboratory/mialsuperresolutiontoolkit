#!/usr/bin/bash

realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

if [[ "$OSTYPE" == "linux-gnu" ]]; then
        # Linux
        DIR="$(dirname $(readlink -f "$0"))"
elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX
        DIR="$(dirname $(realpath $0))"
fi

echo "Building documentation in $DIR/documentation/_build/html"

OLDPWD="$PWD"

cd "$DIR/documentation"
make clean
make html

cd "$OLDPWD"