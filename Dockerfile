FROM ubuntu:14.04

##############################################################
# Ubuntu system setup
##############################################################
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
    bash /tmp/miniconda.sh -bfp /opt/conda && \
    rm -rf /tmp/miniconda.sh && \
    rm -rf /var/lib/apt/lists/*

##############################################################
# Setup and update miniconda
##############################################################
ENV PATH "/opt/conda/bin:$PATH"
RUN conda update conda && \
    conda clean --all --yes

##############################################################
# User/group creation
##############################################################
RUN groupadd -r -g 1000 mialsrtk && \
    useradd -r -M -u 1000 -g mialsrtk mialsrtk

##############################################################
# Copy only code inside the docker image
##############################################################
# Copy only C++ source code
RUN mkdir -p /opt/mialsuperresolutiontoolkit/src
COPY src/ /opt/mialsuperresolutiontoolkit/src/

# Copy PyMIALSRTK code
RUN mkdir -p /opt/mialsuperresolutiontoolkit/pymialsrtk
COPY pymialsrtk/ /opt/mialsuperresolutiontoolkit/pymialsrtk/
COPY setup.py /opt/mialsuperresolutiontoolkit/setup.py
COPY get_version.py /opt/mialsuperresolutiontoolkit/get_version.py

# Copy LICENSE and README files
COPY LICENSE.txt /opt/mialsuperresolutiontoolkit/LICENSE.txt
COPY README.md /opt/mialsuperresolutiontoolkit/README.md
COPY .zenodo.json /opt/mialsuperresolutiontoolkit/.zenodo.json

# Copy docker directories
COPY docker/ /opt/mialsuperresolutiontoolkit/docker/

##############################################################
# Compile C++ MIALSRTK tools
##############################################################
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
# Initialize fake DISPLAY
##############################################################
ENV DISPLAY :0

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
