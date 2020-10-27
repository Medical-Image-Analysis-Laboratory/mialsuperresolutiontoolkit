#! /bin/sh
CMP_BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo $CMP_BUILD_DATE

VERSION=v$(python3 get_version.py)
echo $VERSION

VCS_REF=$(git rev-parse --verify HEAD)
echo $VCS_REF

MAIN_DOCKER="sebastientourbier/mialsuperresolutiontoolkit-ubuntu14.04:${VERSION}"
echo $MAIN_DOCKER

docker build --rm --build-arg BUILD_DATE=$CMP_BUILD_DATE \
				  --build-arg VCS_REF=$VCS_REF \
				  --build-arg VERSION=$VERSION \
				  -t "${MAIN_DOCKER}" . \


docker build --rm --build-arg BUILD_DATE=$CMP_BUILD_DATE \
                             --build-arg VERSION=$VERSION \
                             --build-arg VCS_REF=$VCS_REF \
                             --build-arg MAIN_DOCKER=$MAIN_DOCKER \
                             -t sebastientourbier/mialsuperresolutiontoolkit-bidsapp:${VERSION} ./docker/bidsapp
