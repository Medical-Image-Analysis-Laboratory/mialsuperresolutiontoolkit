FROM ubuntu:14.04
MAINTAINER sebastientourbier
RUN apt-get update && apt-get install software-properties-common -y && apt-add-repository ppa:saiarcot895/myppa -y && apt-get update && apt-get -y install apt-fast \
&& apt-fast install -y \
	build-essential \
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
	python-matplotlib \
&& apt-get clean

# # Git clone the repo from MIALSRTK official repository on GitHub. #
RUN mkdir /opt/mialsrtk-build && cd /opt/mialsrtk-build \ 
&& git clone https://github.com/sebastientourbier/mialsuperresolutiontoolkit.git \
&& cd mialsuperresolutiontoolkit \ 
&& git checkout master && mkdir build

WORKDIR /opt/mialsrtk-build/mialsuperresolutiontoolkit/build

RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -D USE_OMP:BOOL=ON ../src \ 
&& make -j2 && sudo make install && cd .. 

ENV BIN_DIR "/usr/local/bin" 
ENV DISPLAY :0
#&& make install && make clean && cd .. && rm -rf build
