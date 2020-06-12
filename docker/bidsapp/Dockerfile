##############################################################
# Use an initial image, where all MIAL Super-Resolution BIDSApp dependencies are installed, as a parent image
##############################################################
ARG MAIN_DOCKER
FROM $MAIN_DOCKER

##############################################################
# HPC
##############################################################
ENV LANG C.UTF-8 
ENV LC_ALL C.UTF-8 

##############################################################
# Installation of pyMIALSRTK
##############################################################
WORKDIR /app

# Store command related variables
ENV CONDA_ENV_PATH /opt/conda
ENV MY_CONDA_PY3ENV "pymialsrtk-env"
# This is how you will activate this conda environment
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"

# Create the conda environment
ADD environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

# Install pymialsrtk
RUN cd /opt/mialsuperresolutiontoolkit \
    && /bin/bash -c '$CONDA_ACTIVATE && python setup.py install'

# Create entrypoint script
RUN echo '#! /bin/bash \n \
			$CONDA_ACTIVATE \
			&& xvfb-run -a python /opt/mialsuperresolutiontoolkit/pymialsrtk/pipelines/anatomical/srr.py $@' > /app/run_srr.sh

## FOR DEBUG
# RUN echo '#! /bin/bash \n \
# 			echo "User: $USER" \
# 			&& echo "Group:"$(id -g -n $USER) \
# 			&& export && $CONDA_ACTIVATE \
# 			&& xvfb-run -a python /opt/mialsuperresolutiontoolkit/pymialsrtk/pipelines/anatomical/srr.py $@' > /app/run_srr.sh

# Set the working directory back to /app
# Acquire script to be executed
RUN chmod 775 /app/run_srr.sh

# Display all environment variables 
RUN export

##############################################################
# Create input and output directories of the BIDS App
##############################################################
RUN mkdir /bids_dir && \
    chmod -R 777 /bids_dir

RUN mkdir /output_dir && \
    chmod -R 777 /output_dir

##############################################################
# Make singularity happy
##############################################################
RUN ldconfig

##############################################################
# BIDS App entrypoint script
##############################################################
ENTRYPOINT ["/app/run_srr.sh"]

##############################################################
# Build arguments
##############################################################
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

##############################################################
# Metadata
##############################################################
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="MIAL Super-Resolution ToolKit BIDS App"
LABEL org.label-schema.description="Computing environment of the MIAL Super-Resolution BIDS App based on Ubuntu 14.04."
LABEL org.label-schema.url="https://mialsuperresolutiontoolkit.readthedocs.io"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url="https://github.com/sebastientourbier/mialsuperresolutiontoolkit"
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run --rm -v ~/data/bids_dataset:/tmp -t sebastientourbier/mialsuperresolutiontoolkit-ubuntu16.04:${VERSION}"

