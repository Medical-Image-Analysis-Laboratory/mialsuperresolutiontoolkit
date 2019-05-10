#!/bin/sh
#
# Script: Superresolution pipeline for fetal brain MRI
#
# Usage: Run with log saved: sh superresolution_pipeline.sh list_of_scans.txt > reconstruction_original_images.log 
# 
# Author: Sebastien Tourbier
# 
########################################################################################################################

#Tune the number of cores used by the OpenMP library for multi-threading purposes
export OMP_NUM_THREADS=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || sysctl -n hw.ncpu)   

ANAT_DIR=$RESULTS/anat
XFM_DIR=$RESULTS/xfm
RESULTS=$RESULTS/tmp

echo 
echo "-----------------------------------------------"
echo

echo "Starting superresolution pipeline for patient $PATIENT "

echo 
echo "-----------------------------------------------"
echo

#LAMBDA_TV=0.25       #0.75
#DELTA_T=0.1
LOOPS=10
RAD_DILATION=1
START_ITER=1
MAX_ITER=1


echo
echo "Automated brain localization and extraction parameters"
echo
echo "Type of brain mask : ${maskType}"
echo
echo "Super-resolution parameters"
echo
echo "LAMBDA_TV : ${LAMBDA_TV}"
echo "DELTA_T : ${DELTA_T}"
echo "LOOPS : ${LOOPS}"
echo
echo "Brain mask refinement parameters"
echo
echo "Number of loops : ${DELTA_T}"
echo "Morphological dilation radius : ${RAD_DILATION}"
echo 
echo "-----------------------------------------------"
echo

echo "Initialization..."

START1=$(date +%s)

echo 
echo "-----------------------------------------------"
echo

echo "OMP # of cores set to ${OMP_NUM_THREADS}!"
echo

#export BIN_DIR="/usr/local/bin"
printf "BIN_DIR=${BIN_DIR} \n"

SCANS="${1}"
echo "List of scans : $SCANS"


echo "Everything set!"
echo 
echo "-----------------------------------------------"
echo

echo 
echo "-----------------------------------------------"
echo
echo "Should do brain localization and extraction here, but need to make a selection of the best scans with the best brain masks before reconstruction."
echo 
echo "-----------------------------------------------"
echo


VOLS=0
while read -r line
do
	VOLS=$((VOLS+1))
done < "$SCANS"

echo 
echo "-----------------------------------------------"
echo
echo "Number of scans : ${VOLS}"
echo 
echo "-----------------------------------------------"
echo

##Iteration of motion estimation / reconstruction / brain mask refinement
ITER="${START_ITER}"
#for (( ITER=$START_ITER; ITER<=$MAX_ITER; ITER++ ))
while [ "$ITER" -le "$MAX_ITER" ]
do
	echo "Performing iteration # ${ITER}"

	cmdIntensity="mialsrtkIntensityStandardization"
	cmdIntensityNLM="mialsrtkIntensityStandardization"
	cmdHistogramNormalization="python ${BIN_DIR}/mialsrtkHistogramNormalization.py"
	cmdHistogramNormalizationNLM="python ${BIN_DIR}/mialsrtkHistogramNormalization.py"

	if [ "$ITER" -eq "1" ];
	then

		while read -r line
		do
			set -- $line
			stack=$1
			orientation=$2

			echo "Process stack $stack with $orientation orientation..."

		    #Reorient the image
			mialsrtkOrientImage -i ${PATIENT_DIR}/${stack}.nii.gz -o $RESULTS/${stack}_reo_iteration_${ITER}.nii.gz -O "$orientation"
			mialsrtkOrientImage -i ${PATIENT_MASK_DIR}/${stack}_desc-brain_mask.nii.gz -o $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -O "$orientation"

			#denoising on reoriented images
			weight="0.1"
			btkNLMDenoising -i "$RESULTS/${stack}_reo_iteration_${ITER}.nii.gz" -o "$RESULTS/${stack}_nlm_reo_iteration_${ITER}.nii.gz" -b $weight

			#Make slice intensities uniform in the stack
			mialsrtkCorrectSliceIntensity "$RESULTS/${stack}_nlm_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_nlm_uni_reo_iteration_${ITER}.nii.gz"
			mialsrtkCorrectSliceIntensity "$RESULTS/${stack}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_uni_reo_iteration_${ITER}.nii.gz"

			#bias field correction slice by slice
			mialsrtkSliceBySliceN4BiasFieldCorrection "$RESULTS/${stack}_nlm_uni_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_nlm_n4bias.nii.gz"
			mialsrtkSliceBySliceCorrectBiasField "$RESULTS/${stack}_uni_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_nlm_n4bias.nii.gz" "$RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz"

			mialsrtkCorrectSliceIntensity "$RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			mialsrtkCorrectSliceIntensity "$RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz"

			#Intensity rescaling cmd preparation
			cmdIntensityNLM="$cmdIntensityNLM -i $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdIntensity="$cmdIntensity -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdHistogramNormalization="$cmdHistogramNormalization -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz -m  $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
			cmdHistogramNormalizationNLM="$cmdHistogramNormalizationNLM -i $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz -m  $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"

		done < "$SCANS"

		echo "$cmdIntensity"

	else

		while read -r line
		do
			set -- $line
			stack=$1

			#Make slice intensities uniform in the stack
			mialsrtkCorrectSliceIntensity "$RESULTS/${stack}_nlm_reo_iteration_1.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_nlm_uni_reo_iteration_${ITER}.nii.gz"
			mialsrtkCorrectSliceIntensity "$RESULTS/${stack}_reo_iteration_1.nii.gz" "$RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_uni_reo_iteration_${ITER}.nii.gz"

			cmdCorrectBiasField="mialsrtkCorrectBiasFieldWithMotionApplied -i $RESULTS/${stack}_nlm_uni_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -o $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --input-bias-field $RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}_gbcorrfield.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --output-bias-field $RESULTS/${stack}_nlm_n4bias_iteration_${ITER}.nii.gz" 
			cmdCorrectBiasField="$cmdCorrectBiasField -t $RESULTS/${stack}_transform_${VOLS}V_${LAST_ITER}.txt"
			eval "$cmdCorrectBiasField"

			cmdCorrectBiasField="mialsrtkCorrectBiasFieldWithMotionApplied -i $RESULTS/${stack}_uni_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -o $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --input-bias-field $RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}_gbcorrfield.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --output-bias-field $RESULTS/${stack}_n4bias_iteration_${ITER}.nii.gz" 
			cmdCorrectBiasField="$cmdCorrectBiasField -t $RESULTS/${stack}_transform_${VOLS}V_${LAST_ITER}.txt"
			eval "$cmdCorrectBiasField"

			cmdIntensityNLM="$cmdIntensityNLM -i $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdIntensity="$cmdIntensity -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			
			cmdHistogramNormalization="$cmdHistogramNormalization -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}.nii.gz -m  $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
			cmdHistogramNormalizationNLM="$cmdHistogramNormalizationNLM -i $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz -m  $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"

		done < "$SCANS"

	fi

	
	#Intensity rescaling
	eval "$cmdIntensityNLM"
	eval "$cmdIntensity"

	#histogram normalization - need to change the brain mask name expected according to the one used (full auto/localization and rigid extraction/localization only/manual)
	eval "$cmdHistogramNormalization"
	eval "$cmdHistogramNormalizationNLM"

	cmdIntensity="mialsrtkIntensityStandardization"
	cmdIntensityNLM="mialsrtkIntensityStandardization"
	while read -r line
	do
		set -- $line
		stack=$1
		#Intensity rescaling cmd preparation
		cmdIntensityNLM="$cmdIntensityNLM -i $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz -o $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdIntensity="$cmdIntensity -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz -o $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
	done < "$SCANS"

	#Intensity rescaling
	eval "$cmdIntensityNLM"
	eval "$cmdIntensity"

	echo "Initialize the super-resolution image using initial masks - Iteration ${ITER}..."

	cmdImageRECON="mialsrtkImageReconstruction --mask"
	cmdSuperResolution="mialsrtkTVSuperResolution"
	#cmdRobustSuperResolution="$MIALSRTK_APPLICATIONS/mialsrtkRobustTVSuperResolutionWithGMM"

	#Preparation for (1) motion estimation and SDI reconstruction and (2) super-resolution reconstruction
	while read -r line
	do
		set -- $line
		stack=$1
		mialsrtkMaskImage -i "$RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz" -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -o "$RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"

		cmdImageRECON="$cmdImageRECON -i $RESULTS/${stack}_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdImageRECON="$cmdImageRECON -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz"
		cmdImageRECON="$cmdImageRECON -t $RESULTS/${stack}_transform_${VOLS}V_${ITER}.txt"

		cmdSuperResolution="$cmdSuperResolution -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdSuperResolution="$cmdSuperResolution -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz"
		cmdSuperResolution="$cmdSuperResolution -t $RESULTS/${stack}_transform_${VOLS}V_${ITER}.txt"
		#cmdRobustSuperResolution="$cmdRobustSuperResolution -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz -t $RESULTS/${stack}_transform_${VOLS}V_${ITER}.txt"
	done < "$SCANS"

	#Run motion estimation and SDI reconstruction
	echo "Run motion estimation and scattered data interpolation - Iteration ${ITER}..."

	cmdImageRECON="$cmdImageRECON -o $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz"
	eval "$cmdImageRECON"

	echo "Done"
	echo
	echo "##########################################################################################################################"
	echo

	#Brain image super-reconstruction
	echo "Reconstruct the super-resolution image with initial brain masks- Iteration ${ITER}..."

	cmdSuperResolution="$cmdSuperResolution -o $RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz" 
	cmdSuperResolution="$cmdSuperResolution -r $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz" 
	cmdSuperResolution="$cmdSuperResolution --bregman-loop 1 --loop ${LOOPS} --iter 50 --step-scale 10 --gamma 10 --deltat ${DELTA_T}" 
	cmdSuperResolution="$cmdSuperResolution --lambda ${LAMBDA_TV} --inner-thresh 0.00001 --outer-thresh 0.000001"
	echo "CMD: $cmdSuperResolution"
	eval "$cmdSuperResolution"

	echo "Done"
	echo

	#up="1.0"
	#cmdRobustSuperResolution="$cmdRobustSuperResolution -o $RESULTS/RobustSRTV_${PATIENT}_${VOLS}V_NoNLM_bcorr_norm_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_it${ITER}_rad${RAD_DILATION}_up${up}.nii.gz -r $RESULTS/SDI_${PATIENT}_${VOLS}V_nlm_bcorr_norm_it${ITER}_rad${RAD_DILATION}.nii.gz --bregman-loop 1 --loop ${LOOPS} --iter 50 --step-scale 10 --gamma 10 --lambda ${LAMBDA_TV} --deltat ${DELTA_T} --inner-thresh 0.00001 --outer-thresh 0.000001 --use-robust --huber-mode 2 --upscaling-factor $up"
	#eval "$cmdRobustSuperResolution"

	NEXT_ITER=${ITER}
	NEXT_ITER=$((NEXT_ITER+1))

	echo "##########################################################################################################################"
	echo

	echo "Refine the mask of the HR image for next iteration (${NEXT_ITER})..."

	echo
	echo "##########################################################################################################################"
	echo

	#Preparation for brain mask refinement
	cmdRefineMasks="mialsrtkRefineHRMaskByIntersection --use-staple --radius-dilation ${RAD_DILATION}"
	while read -r line
	do
		set -- $line
		stack=$1
		cmdRefineMasks="$cmdRefineMasks -i $RESULTS/${stack}_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdRefineMasks="$cmdRefineMasks -m $RESULTS/${stack}_desc-brain_mask_reo_iteration_${ITER}.nii.gz"
		cmdRefineMasks="$cmdRefineMasks -t $RESULTS/${stack}_transform_${VOLS}V_${ITER}.txt"
		cmdRefineMasks="$cmdRefineMasks -O $RESULTS/${stack}_desc-brain_mask_reo_iteration_${NEXT_ITER}.nii.gz"
	done < "$SCANS"

	#Brain mask refinement

	cmdRefineMasks="$cmdRefineMasks -o $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_brain_mask.nii.gz -r $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz"
	eval "$cmdRefineMasks"

	#Bias field refinement
	mialsrtkN4BiasFieldCorrection "$RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz" "$RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_brain_mask.nii.gz" "$RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_gbcorr.nii.gz" "$RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_gbcorrfield.nii.gz"

	#Brain masking of the reconstructed image

	mialsrtkMaskImage -i "$RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz" -m "$RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_brain_mask.nii.gz" -o $RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_masked.nii.gz

	echo
	echo "##########################################################################################################################"
	echo

	LAST_ITER="$ITER"
	ITER=$((ITER+1))

done
LAST_ITER=1


# Extract absolute path of the folder containing the scripts (i.e. /fetaldata/code)
CODE_DIR="$(dirname "$0")"
CODE_DIR="$(readlink -f $CODE_DIR)"

echo "Dataset code directory: ${CODE_DIR}"

# Move the preprocessed input scans to anat and their associated slice-to-volume transform and 
# create their respective json file (BIDS Common Derivatives RC1)

SOURCES=""
while read -r line
do
	set -- $line
	stack=$1

	# Copy and rename the transform and the image
	cp "$RESULTS/${stack}_transform_${VOLS}V_${LAST_ITER}.txt" "${XFM_DIR}/${stack}_from-orig_to-SDI_mode-image_xfm.txt"
	cp "$RESULTS/${stack}_uni_bcorr_reo_iteration_${LAST_ITER}_histnorm.nii.gz" "${ANAT_DIR}/${stack}_preproc.nii.gz"

	# Create the BIDS json sidecar
	sh ${CODE_DIR}/create_scan_preproc_json.sh "${ANAT_DIR}/${stack}_preproc.json" "${PATIENT_DIR}/${stack}.nii.gz" 

	SOURCES="${SOURCES} \"${ANAT_DIR}/${stack}_preproc.nii.gz\", \"${XFM_DIR}/${stack}_from-orig_to-SDI_mode-image_xfm.txt\" ,"

done < "$SCANS"

# Move the reconstructed images to anat and create the json file (BIDS Common Derivatives RC1)

ANAT_DIR="$(dirname "$RESULTS")/anat"
echo "Copy final outputs to ${ANAT_DIR}"
cp "$RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${LAST_ITER}.nii.gz" "${ANAT_DIR}/${PATIENT}_rec-SDI_T2w.nii.gz"
cp "$RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${LAST_ITER}.nii.gz" "${ANAT_DIR}/${PATIENT}_rec-SR_T2w.nii.gz"
cp "$RESULTS/SRTV_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${LAST_ITER}_masked.nii.gz" "${ANAT_DIR}/${PATIENT}_rec-SR_desc-masked_T2w.nii.gz"


# Create the json file for the super-resolution images
OUTPUT_JSON="${ANAT_DIR}/${PATIENT}_rec-SR.json"
(
cat <<EOF
{
	"Description": "Isotropic high-resolution image reconstructed using the Total-Variation Super-Resolution algorithm provided by MIALSRTK",
    "Sources": [${SOURCES}],
    "CustomMetaData": {
    	"Number of scans used":   $VOLS,
		"TV regularization weight lambda": ${LAMBDA_TV},
		"Optimization time step":   ${DELTA_T},
		"Primal/dual loops": $LOOPS,
		"Number of pipeline iterations" : ${LAST_ITER}
	}
}

EOF
) > $OUTPUT_JSON

echo
echo "##########################################################################################################################"
echo

END1=$(date +%s)

DIFF1=$(( $END1 - $START1 ))

echo "Done. It took $DIFF1 seconds for reconstruction, after $MAX_ITER refinement loops, using ${VOLS} volumes."
echo 
echo "-----------------------------------------------"
echo
