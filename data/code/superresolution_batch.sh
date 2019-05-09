#!/bin/sh
# usage:
# sh generic_script_to_run_superresolution.sh /path/to/batch_list.txt

# Get the directory where the script is stored,
# which is supposed to be in the code folder of the dataset root directory)
SCRIPT_DIR="$(dirname "$0")"

#Get absolute path
SCRIPT_DIR="$(readlink -f $SCRIPT_DIR)"

DATASET_DIR="$(dirname "${SCRIPT_DIR}")"

echo "Dataset directory : $DATASET_DIR"

#from each line of the subject_parameters.txt given as input
while read -r line
  do
  #Extract PATIENT, LAMBDA_TV and DELTA_T and the scan list sub-01_run-01_scans.txt
	set -- $line
	PATIENT=$1
	LAMBDA_TV=$2
  DELTA_T=$3
  SCANS_LIST="${PATIENT}_${4}.txt"

  echo "> Process subject ${PATIENT} with lambda_tv = ${LAMBDA_TV} and delta_t = ${DELTA_T}"

  PATIENT_DIR="$DATASET_DIR/${PATIENT}"
  echo "  ... Subject directory : ${PATIENT_DIR}"

  # Create the output directory for results (if not existing). 
  # run-XX is used to identify different list of scans:
  # sub-01_run-01_scans.txt, sub-01_run-02_scans.txt, ..., sub-01_run-XX_scans.txt
  # here I am just using sub-01_run-01_scans.txt but you can think as I am looping over 
  # the different calls while reading the batch_list.txt file, 
  # you could  similarly loop over different list of scans.
  # The final superesolution is saved in derivatives/mialsrtk_tv-${LAMBDA_TV}_dt-${DELTA_T}_run-01/$PATIENT/anat folder
  # All intermediate outputs are saved in a tmp folder (see below). 
  
  RESULTS="derivatives/mialsrtk_tv-${LAMBDA_TV}_dt-${DELTA_T}_run-01/$PATIENT"
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
              --env RESULTS="/fetaldata/${RESULTS}/tmp" \
              -t sebastientourbier/mialsuperresolutiontoolkit:v1.1.0 \
              "/fetaldata/code/${SCANS_LIST}"

done < "$1"
