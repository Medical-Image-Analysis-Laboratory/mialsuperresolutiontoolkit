#!/bin/sh
# usage:
# sh superresolution_batch.sh /path/to/batch_list.txt
# 
# Author: Sebastien Tourbier
# 
###################################################################

# Use the latest stable release version of the docker image
VERSION_TAG="v1.0.0"

# Get the directory where the script is stored,
# which is supposed to be in the code folder of the dataset root directory)
# and get absolute path
SCRIPT_DIR="$(dirname "$0")"
SCRIPT_DIR="$(readlink -f $SCRIPT_DIR)"

# Get BIDS dataset root directory
DATASET_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Dataset root directory : $DATASET_DIR"



#from each line of the subject_parameters.txt given as input
while read -r line
  do
  #Extract PATIENT, LAMBDA_TV and DELTA_T and the scan list sub-01_ses-01_scans.txt
	set -- $line
	PATIENT=$1
	LAMBDA_TV=$3
  DELTA_T=$4
  RECON_SESSION=$2
  SCANS_LIST="${PATIENT}_${RECON_SESSION}_scans.txt"

  echo "> Process subject ${PATIENT} with lambda_tv = ${LAMBDA_TV} and delta_t = ${DELTA_T}"

  PATIENT_DIR="$DATASET_DIR/${PATIENT}"
  echo "  ... Subject directory : ${PATIENT_DIR}"

  # Get the number of scans in the list
  # Can be used to differentiate output folders
  # i.e. line RESULTS="derivatives/mialsrtk/$PATIENT/${RECON_SESSION}" would become
  # RESULTS="derivatives/mialsrtk_scans-${VOLS}/$PATIENT/${RECON_SESSION}"

  VOLS=0
  while read -r line
  do
    VOLS=$((VOLS+1))
  done < "${DATASET_DIR}/code/${SCANS_LIST}"

  echo "  ... Number of scans : ${VOLS}"
  echo

  # Create the output directory for results (if not existing). 
  # run-XX is used to identify different list of scans:
  # sub-01_ses-01_scans.txt, sub-01_ses-02_scans.txt, ..., sub-01_ses-XX_scans.txt
  # The final superesolution is saved in derivatives/mialsrtk/$PATIENT/ses-XX/anat folder
  # All intermediate outputs are saved in a tmp folder (see below). 
   
  if [ ! -d "${DATASET_DIR}/derivatives/mialsrtk" ]; then
    mkdir -p "${DATASET_DIR}/derivatives/mialsrtk";
    echo "    * Folder ${DATASET_DIR}/derivatives/mialsrtk created"
  fi

  if [ ! -f "${DATASET_DIR}/derivatives/mialsrtk/dataset_description.json" ];
  then
    sh ${DATASET_DIR}/code/create_dataset_description_json.sh "${DATASET_DIR}/derivatives/mialsrtk/dataset_description.json" "$VERSION_TAG"
  fi
  
  RESULTS="derivatives/mialsrtk/$PATIENT/${RECON_SESSION}"
  echo "  ... Reconstruction tmp directory: ${DATASET_DIR}/${RESULTS}"
  if [ ! -d "${DATASET_DIR}/${RESULTS}" ]; then
    mkdir -p "${DATASET_DIR}/${RESULTS}";
    echo "    * Folder ${DATASET_DIR}/${RESULTS} created"
  fi
  if [ ! -d "${DATASET_DIR}/${RESULTS}/tmp" ]; then
    mkdir -p "${DATASET_DIR}/${RESULTS}/tmp";
    echo "    * Folder ${DATASET_DIR}/${RESULTS}/tmp created"
  fi
  if [ ! -d "${DATASET_DIR}/${RESULTS}/anat" ]; then
    mkdir -p "${DATASET_DIR}/${RESULTS}/anat";
    echo "    * Folder ${DATASET_DIR}/${RESULTS}/anat created"
  fi
  if [ ! -d "${DATASET_DIR}/${RESULTS}/xfm" ]; then
    mkdir -p "${DATASET_DIR}/${RESULTS}/xfm";
    echo "    * Folder ${DATASET_DIR}/${RESULTS}/xfm created"
  fi

  PATIENT_MASK_DIR="derivatives/manual_masks/$PATIENT"
  

  # Run the super-resolution pipeline

  docker run --rm -u $(id -u):$(id -g) \
              -v $DATASET_DIR:/fetaldata \
              --entrypoint /fetaldata/code/superresolution_pipeline.sh \
              --env PATIENT="$PATIENT" \
              --env DELTA_T="$DELTA_T" \
              --env LAMBDA_TV="$LAMBDA_TV "\
              --env PATIENT_DIR="/fetaldata/${PATIENT}" \
              --env PATIENT_MASK_DIR="/fetaldata/${PATIENT_MASK_DIR}/anat" \
              --env RESULTS="/fetaldata/${RESULTS}" \
              -t sebastientourbier/mialsuperresolutiontoolkit:"$VERSION_TAG" \
              "/fetaldata/code/${SCANS_LIST}"



done < "$1"
