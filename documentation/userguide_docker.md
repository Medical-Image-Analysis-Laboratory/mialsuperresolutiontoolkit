![MIALSRTK logo](https://cloud.githubusercontent.com/assets/22279770/24004342/5e78836a-0a66-11e7-8b7d-058961cfe8e8.png)

Copyright © 2016-2017 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland 

---

# Instructions for running Docker image #

```
Note: tested on Ubuntu 16.04

```
## Docker installation
Download and Install Docker from [Website](https://www.docker.com/get-docker) (PC,MAC,LINUX) OR with apt-get (Debian/Ubuntu):

```
#!bash

sudo apt-get install docker docker-compose docker-registry docker.io python-docker python-dockerpty

```
## To run lastest version of MIALSRTK docker  image 

1) Make sure no docker mialsuperresolutiontoolkit container are already running

```
#!bash

sudo docker ps

```
2) If a container is already running, identify its iD (CID) from the previous command results and remove it

```
#!bash

sudo docker rm -f CID

```
3) Make sure no other docker image of mialsrtk are stored

```
#!bash

sudo docker images

```
4) If an image is already existing, identify from the previous command results the image iD (IID) and remove it

```
#!bash

sudo docker rmi -f IID

```
5) Retrieve the MIALSRTK docker image at its latest vesion

```
#!bash

sudo docker pull sebastientourbier/mialsuperresolutiontoolkit

```
6) Run the docker image

```
#!bash

sudo docker run -it sebastientourbier/mialsuperresolutiontoolkit

```
## For testing

1) Go to data folder

```
#!bash

cd ../data/

```
2) Run super-resolution pipeline

```
#!bash

sh superresolution_autoloc.sh listScansRECONauto.txt

```
## For your own data

1) Prepare your local volume to be mounted on docker

2) Run the docker image with local volume (/home/localadmin/Desktop/Jakab) mounted (as /fetaldata) 

```
#!bash

sudo docker run -v /home/localadmin/Desktop/Jakab:/fetaldata -it sebastientourbier/mialsuperresolutiontoolkit

```
3) Go to mounted volume

```
#!bash

cd /fetaldata

```
4) Run super-resolution pipeline

```
#!bash
sh superresolution_autoloc.sh listScansRECONauto.txt

```

## To build the docker image 

1) Clone the github repository of mialsrtk to your <INSTALLATION DIRECTORY>
```console
$ cd <INSTALLATION DIRECTORY>
$ git clone https://github.com/sebastientourbier/mialsuperresolutiontoolkit.git
$ cd mialsuperresolutiontoolkit
$ (sudo) docker build --rm (--no-cache) -f docker/Dockerfile -t sebastientourbier/mialsuperresolutiontoolkit:<YourTag> .
```
---


# Contact #

* Sébastien Tourbier - sebastien(dot)tourbier1(at)gmail(dot)com

---
