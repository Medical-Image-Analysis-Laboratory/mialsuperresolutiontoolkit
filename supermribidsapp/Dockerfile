FROM sebastientourbier/mialsuperresolutiontoolkit:v1.0

ENV CONDA_ENV supermri-env

RUN /bin/bash -c ". activate $CONDA_ENV && \
    python setup.py install"

WORKDIR /tmp

ENTRYPOINT ["/opt/mialsrtk-build/mialsuperresolutiontoolkit/supermribidsapp/scripts/run.py"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

#Metadata
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="Super Resolution MRI BIDS App"
LABEL org.label-schema.description="Super Resolution MRI BIDS App - the next generation of the MIAL Super-Resolution Toolkit"
LABEL org.label-schema.url="https://supermribidsapp.readthedocs.io"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url=""
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Connectomics Lab, Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run -v ~/data/bids_dataset:/tmp"
