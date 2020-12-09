##############################################################
# Use an initial image where all MIAL Super-Resolution BIDSApp
# dependencies are installed, as a parent image
##############################################################
ARG MAIN_DOCKER
FROM "${MAIN_DOCKER}"

##############################################################
# HPC
##############################################################
ENV LANG C.UTF-8 
ENV LC_ALL C.UTF-8 

##############################################################
# Create input and output directories of the BIDS App
##############################################################
RUN mkdir /bids_dir && \
    chmod -R 777 /bids_dir

RUN mkdir /output_dir && \
    chmod -R 777 /output_dir

##############################################################
# Create .cache and set right permissions for generated
# Python egg cache
##############################################################
RUN mkdir /.cache && \
    chmod -R 777 /.cache

##############################################################
# Installation of pyMIALSRTK
##############################################################
# Set the working directory to /app
WORKDIR /app

# Store command related variables
ENV CONDA_ENV_PATH /opt/conda
ENV MY_CONDA_PY3ENV "pymialsrtk-env"
# This is how you will activate this conda environment
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"

# Create the conda environment
COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

# Set the working directory to /opt/mialsuperresolutiontoolkit and install pymialsrtk
WORKDIR /opt/mialsuperresolutiontoolkit
RUN /bin/bash -c 'ls -al && $CONDA_ACTIVATE && pip install .'

# Set the working directory back to /app
WORKDIR /app

##############################################################
# Use tcmalloc instead of malloc in TensorFLow
# that suffers less from fragmentation when
# allocating and deallocating large objects
##############################################################
RUN apt-get update && apt-get install -y google-perftools && \
    rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD=/usr/lib/libtcmalloc.so.4

##############################################################
# Create entrypoint scripts
##############################################################
# # Create entrypoint script
# RUN echo '#! /bin/bash \n \
# 			$CONDA_ACTIVATE \
# 			&& xvfb-run -a python /opt/mialsuperresolutiontoolkit/docker/bidsapp/run.py $@' > /app/run_srr.sh

# # Create entrypoint script
# RUN echo '#! /bin/bash \n \
# 			$CONDA_ACTIVATE \
# 			&& xvfb-run -a coverage run --source=pymialsrtk \
# 			/opt/mialsuperresolutiontoolkit/docker/bidsapp/run.py $@\
# 			&& coverage html -d /bids_dir/code/coverage_html' > /app/run_srr_coverage.sh

# Create and write content to BIDSapp entrypoint script
RUN export content="echo User: \$(id -un \$USER) && echo Group: \$(id -gn \$USER) &&" && \
    export content="${content} . activate \"${MY_CONDA_PY3ENV}\" &&" && \
    export content="${content} xvfb-run -a python /opt/mialsuperresolutiontoolkit/docker/bidsapp/run.py \$@" && \
    printf "$content" > /app/run_srr.sh \
    && unset content
RUN cat /app/run_srr.sh

# Create  and write content to BIDSapp entrypoint script
RUN export content_cov="echo User: \$(id -un \$USER) && echo Group: \$(id -gn \$USER) &&" && \
    export content_cov="${content_cov} . activate \"${MY_CONDA_PY3ENV}\" &&" && \
    export content_cov="${content_cov} xvfb-run -a coverage run --source=pymialsrtk \
/opt/mialsuperresolutiontoolkit/docker/bidsapp/run.py \$@ \
|& tee /bids_dir/code/log.txt &&" && \
    export content_cov="${content_cov} coverage html -d /bids_dir/code/coverage_html &&" && \
    export content_cov="${content_cov} coverage xml -o /bids_dir/code/coverage.xml &&" && \
    export content_cov="${content_cov} coverage json -o /bids_dir/code/coverage.json" && \
    printf "$content_cov" > /app/run_srr_coverage.sh \
    && unset content_cov
RUN cat /app/run_srr_coverage.sh

##############################################################
# Acquire script to be executed
##############################################################
RUN chmod 775 /app/run_srr.sh
RUN chmod 775 /app/run_srr_coverage.sh

##############################################################
# Display all environment variables
##############################################################
# RUN export

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
LABEL org.label-schema.description="Docker image of the MIAL Super-Resolution BIDS App based on Ubuntu 14.04."
LABEL org.label-schema.url="https://mialsrtk.readthedocs.io"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit"
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run -t --rm -u $(id -u):$(id -g) -v /home/localadmin/data/ds001:/bids_dir -v /media/localadmin/data/ds001/derivatives:/output_dir sebastientourbier/mialsuperresolutiontoolkit-bidsapp:${VERSION} /bids_dir /output_dir participant --participant_label 01 --param_file /bids_dir/code/participants_params.json (--openmp_nb_of_cores 4) (--nipype_nb_of_cores 4)"
