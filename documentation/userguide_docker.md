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

A sample dataset is provided along with the code in the data/ folder and available in the docker image. This dataset is structured following the [Brain Imaging Data Structure (BIDS) standard](https://bids-specification.readthedocs.io/en/stable/).

1) Go to data/code folder

```
#!bash

cd ../data/code

```
2) Run super-resolution pipeline

```
#!bash

sh superresolution_batch.sh sub-01_run-01_scans.txt

```
## For your own data

1) Prepare your local volume (dataset root directory) to be mounted on docker. The dataset should be structured following the Brain Imaging Data Structure standard. The input dataset is required to be in valid [Brain Imaging Data Structure (BIDS)]() format, and it must include at least one T2w image with anisotropic resolution per acquisition direction. We highly recommend that you (1) start with the sample dataset, (2) update it with your data, and (3) validate your dataset with the free, [online BIDS Validator](http://bids-standard.github.io/bids-validator/).

2) Then, you can run the docker image in two different ways:

	1) Single subject processing 

	To perform the superresolution pipeline on sub-01, run the docker image with local volume (/home/localadmin/Desktop/Jakab) mounted (as /fetaldata) 

	```console
	$ PATIENT='sub-01'
	$docker run --rm -u $(id -u):$(id -g) \
	              -v <Local/Path/to/your/BIDSdataset>:/fetaldata \
	              --entrypoint /fetaldata/code/superresolution_pipeline.sh \
	              --env PATIENT="$PATIENT" \
	              --env DELTA_T="0.01" \
	              --env LAMBDA_TV="0.75"\
	              --env PATIENT_DIR="/fetaldata/${PATIENT}" \
	              --env PATIENT_MASK_DIR="/fetaldata/derivatives/manual_masks/${PATIENT}/anat" \
	              --env RESULTS="/fetaldata/derivatives/mialsrtk/${PATIENT}/tmp" \
	              -t sebastientourbier/mialsuperresolutiontoolkit:v1.1.0 \
	              "/fetaldata/code/sub-01_run-01_scans.txt"

	```

	2) Batch processing

	A script called `superresolution_batch.sh`, a text file called `batch_list.txt`, and a text file called `sub-01_run-01_scans.txt` are provided in the sample dataset. The script `superresolution_batch.sh` takes the `batch_list.txt` text file as input, where each line is corresponding to a specific run for which the first element is the subject name (sub-01), the second element is the regularization weight lambda, the third element is the optimization time step, and the fourth element is the name of the text used to list the scans without any extension (for instance sub-01_run-01_scans). Each line of the list of scans `sub-01_run-01_scans.txt` corresponds to the description of one input scan for which the first element is the name of the scan without any extension (for instance sub-01_run-01_T2w) and the second element is the scan orientation (possible values are {axial, coronal, sagittal})


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
