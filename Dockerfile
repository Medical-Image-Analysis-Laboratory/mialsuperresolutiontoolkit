FROM ubuntu:14.04

##############################################################
# Ubuntu system setup
##############################################################
ENV CONDA_ENV_PATH /opt/conda
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    apt-add-repository ppa:saiarcot895/myppa -y && \
    apt-get update && \
    apt-get -y install apt-fast \
    && apt-fast install -y \
    build-essential \
    exfat-fuse \
    exfat-utils \
    npm \
    curl \
    bzip2 \
    xvfb \
    x11-apps \
    git \
    gcc-4.8 \
        g++-4.8 \
        cmake \
        libtclap-dev \
        libinsighttoolkit4.5 \
        libinsighttoolkit4-dev \
        libvtk5-dev \
        libvtk5-qt4-dev \
        libvtk5.8 \
        libvtk5.8-qt4 \
        tcl-vtk \
        libvtk-java \
        python-vtk \
        python-vtkgdcm \
        libncurses5  \
        libncurses5-dev \
    libann-dev && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp "$CONDA_ENV_PATH" && \
    rm -rf /tmp/miniconda.sh && \
    rm -rf /var/lib/apt/lists/*

##############################################################
# Setup and update miniconda
##############################################################
ENV PATH "$CONDA_ENV_PATH/bin:$PATH"
RUN conda update conda && \
    conda clean --all --yes

##############################################################
# User/group creation
##############################################################
RUN groupadd -r -g 1000 mialsrtk && \
    useradd -r -M -u 1000 -g mialsrtk mialsrtk

##############################################################
# Copy and compile C++ MIALSRTK code
##############################################################
# Copy only C++ source code
RUN mkdir -p /opt/mialsuperresolutiontoolkit/src
COPY src/ /opt/mialsuperresolutiontoolkit/src/

# Create the build directory and set the working directory
# to this directory
WORKDIR /opt/mialsuperresolutiontoolkit
RUN mkdir build
WORKDIR /opt/mialsuperresolutiontoolkit/build

# Configure and compile C++ MIALSRTK tools
# You can increase the number of cores used by make ("make -jN")
# to speed up local build. However, make sure that it is
# set back to make -j2 before pushing any change to GitHub.
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D USE_OMP:BOOL=ON ../src \
    && make -j6 && make install

##############################################################
# Make MIALSRTK happy
##############################################################
ENV BIN_DIR "/usr/local/bin"
ENV PATH "${BIN_DIR}:$PATH"

##############################################################
# Python cache setup and creation of conda environment
##############################################################
# Create .cache and set right permissions for generated
# Python egg cache
RUN mkdir /.cache && \
    chmod -R 777 /.cache

# Set the working directory to /app
WORKDIR /app
RUN chmod -R 777 /app

# Store command related variables
ENV MY_CONDA_PY3ENV "pymialsrtk-env"
# This is how you will activate this conda environment
ENV CONDA_ACTIVATE "source $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV"

# Create the conda environment
COPY docker/bidsapp/environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

##############################################################
# Setup for scikit-image
#
# Commented for now as it causes issues with Singularity
# for scikit-image = 0.18.3 with OSError:
# [Errno 30] Read-only file system: '/app/skimage/0.18.3/data'
#
# The creation of the datadir for skimage has been introduced
# in 0.17 and so prior versions such as 0.16.2 should not
# perform this process, the version that is now used.
#
# See https://github.com/scikit-image/scikit-image/issues/4664
# for reference.
#
# Similar error encountered for:
#   - fmriprep: https://github.com/nipreps/fmriprep/issues/1777
#   - mriqc: https://neurostars.org/t/read-only-error-in-mriqc-using-singularity-on-cluster/2022
#
##############################################################
# ENV SKIMAGE_VERSION "0.18.3"
# ENV SKIMAGE_DATADIR "/app/skimage"
# RUN mkdir -p "${SKIMAGE_DATADIR}/${SKIMAGE_VERSION}" && \
#     chmod -R 777 "${SKIMAGE_DATADIR}/${SKIMAGE_VERSION}"
# RUN . $CONDA_ENV_PATH/bin/activate $MY_CONDA_PY3ENV && \
#     python -c "import skimage"

##############################################################
# Setup for tensorflow
##############################################################
# Filter out all messages
# ENV TF_CPP_MIN_LOG_LEVEL "0"

# Make tensorflow happy: Use jemalloc instead of malloc.
# Jemalloc suffers less from fragmentation when allocating
# and deallocating large objects
RUN apt-get update && apt-get install -y libjemalloc-dev && \
    rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so

# Use tcmalloc instead of malloc in TensorFLow
# that suffers less from fragmentation when
# allocating and deallocating large objects
# RUN apt-get update && apt-get install -y google-perftools && \
#     rm -rf /var/lib/apt/lists/*
# ENV LD_PRELOAD=/usr/lib/libtcmalloc.so.4

##############################################################
# Initialize fake DISPLAY
##############################################################
ENV DISPLAY :0

##############################################################
# Copy the rest of the files (Pymialsrtk, license, readme and
# bidsapp entrypoint script) at the end to prevent recompiling
# again the C++ code even if no change was introduced
##############################################################

# Copy PyMIALSRTK code
RUN mkdir -p /opt/mialsuperresolutiontoolkit/pymialsrtk
COPY pymialsrtk/ /opt/mialsuperresolutiontoolkit/pymialsrtk/
COPY setup.py /opt/mialsuperresolutiontoolkit/setup.py
COPY get_version.py /opt/mialsuperresolutiontoolkit/get_version.py

##############################################################
# Copy LICENSE and README and .zenodo.json contributors files
##############################################################
COPY LICENSE.txt /opt/mialsuperresolutiontoolkit/LICENSE.txt
COPY README.md /opt/mialsuperresolutiontoolkit/README.md
COPY .zenodo.json /opt/mialsuperresolutiontoolkit/.zenodo.json

##############################################################
# Arguments passed to the docker build command
##############################################################
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

##############################################################
# Metadata
##############################################################
LABEL org.label-schema.build-date="$BUILD_DATE"
LABEL org.label-schema.name="MIAL Super-Resolution ToolKit Ubuntu 14.04"
LABEL org.label-schema.description="Computing environment of the MIAL Super-Resolution BIDS App based on Ubuntu 14.04."
LABEL org.label-schema.url="https://mialsrtk.readthedocs.io"
LABEL org.label-schema.vcs-ref="$VCS_REF"
LABEL org.label-schema.vcs-url="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit"
LABEL org.label-schema.version="$VERSION"
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run --rm -v ~/data/bids_dataset:/tmp -t sebastientourbier/mialsuperresolutiontoolkit-ubuntu16.04:${VERSION}"
