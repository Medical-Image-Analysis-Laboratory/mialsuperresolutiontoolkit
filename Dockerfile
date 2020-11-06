FROM ubuntu:14.04

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
    libann-dev \
    python-qt4 \
    python-nibabel \
    python-numpy \
    python-scipy \
    python-matplotlib && \
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -bfp /opt/conda && \
    rm -rf /tmp/miniconda.sh && \
    apt-get clean


ENV PATH /opt/conda/bin:$PATH

RUN conda update conda && \
    conda clean --all --yes

RUN groupadd -r -g 1000 mialsrtk && \
    useradd -r -M -u 1000 -g mialsrtk mialsrtk

WORKDIR /opt/mialsuperresolutiontoolkit

COPY . /opt/mialsuperresolutiontoolkit

RUN mkdir build
WORKDIR /opt/mialsuperresolutiontoolkit/build

RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D USE_OMP:BOOL=ON ../src \
    && make -j2 && make install

# Make MIALSRTK happy
ENV BIN_DIR "/usr/local/bin" 
ENV PATH ${BIN_DIR}:$PATH

ENV DISPLAY :0

ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

#Metadata
LABEL org.label-schema.build-date=$BUILD_DATE
LABEL org.label-schema.name="MIAL Super-Resolution ToolKit Ubuntu 14.04"
LABEL org.label-schema.description="Computing environment of the MIAL Super-Resolution BIDS App based on Ubuntu 14.04."
LABEL org.label-schema.url="https://mialsuperresolutiontoolkit.readthedocs.io"
LABEL org.label-schema.vcs-ref=$VCS_REF
LABEL org.label-schema.vcs-url="https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit"
LABEL org.label-schema.version=$VERSION
LABEL org.label-schema.maintainer="Sebastien Tourbier <sebastien.tourbier@alumni.epfl.ch>"
LABEL org.label-schema.vendor="Centre Hospitalier Universitaire Vaudois (CHUV), Lausanne, Switzerland"
LABEL org.label-schema.schema-version="1.0"
LABEL org.label-schema.docker.cmd="docker run --rm -v ~/data/bids_dataset:/tmp -t sebastientourbier/mialsuperresolutiontoolkit-ubuntu16.04:${VERSION}"

