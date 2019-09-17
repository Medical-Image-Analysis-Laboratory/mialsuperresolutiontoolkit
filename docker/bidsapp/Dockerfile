# Use an initial image, where all MIAL Super-Resolution BIDSApp dependencies are installed, as a parent image

ARG MAIN_DOCKER
FROM $MAIN_DOCKER

# Pull the environment name out of the environment.yml
ADD environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

# Install pymialsrtk inside installed conda environment (see environment.yml)
#RUN /bin/bash -c ". activate pymialsrtk-env && python setup.py install"

#Temporary tmp folder (if datalad and singularity used in the future)
RUN /bin/bash -c "mkdir -p /var/tmp"
ENV TMPDIR /var/tmp
ENV TMP /var/tmp
ENV TEMP /var/tmp

# Make it work under singularity (source: https://brainlife.io/docs/apps/container/)
RUN ldconfig

#ENTRYPOINT ["/opt/mialsuperresolutiontoolkit/run.py"]

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

#Metadata
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="MIAL Super-Resolution BIDS App"
LABEL org.label-schema.description="BIDS App performing..."
LABEL org.label-schema.url="https://mialsuperresolutiontoolkit.readthedocs.io"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url="https://github.com/sebastientourbier/mialsuperresolutiontoolkit"
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run --rm -v ~/data/bids_dataset:/tmp -t sebastientourbier/mialsuperresolutionbidsapp:${VERSION}"

