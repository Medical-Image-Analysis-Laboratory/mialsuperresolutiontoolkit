#! Script: Brain image reconstruction / Brain superresolution image using BTK
# Run with log saved: sh reconstruction.sh > reconstruction_original_images.log 
#Tune the number of cores used by the OpenMP library for multi-threading purposes
export OMP_NUM_THREADS=8

PATIENT=$(basename "$PWD")
GA=30

echo 
echo "-----------------------------------------------"
echo

echo "Processing patient $PATIENT (GA=${GA})"

echo 
echo "-----------------------------------------------"
echo

LAMBDA_TV=0.25       #0.75
DELTA_T=0.1
LOOPS=10
RAD_DILATION=1
START_ITER=1
MAX_ITER=3
#maskType="_SSMMI_CompositeVersor2DBSplineNCC" # masks after automatic template-based localization and deformable slice-to-template extraction
#maskType="_SSMMI_VersorOnlyNCC" # masks after automatic template-based localization and rigid slice-to-template extraction
#maskType="" # masks after automatic template-based localization
maskType="" # manually drawn masks

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


export MIALSRTK_PREFIX=/home/tourbier/Desktop/externalmialtoolkit/bin
export SLICER_DIR=/home/tourbier/Softwares/Slicer-4.4.0-linux-amd64/lib/Slicer-4.4/cli-modules
export ATLAS_DIR=/media/Shared_Data_/MyStudies/Atlas

#export DIR_PREFIX=$(dirname "$0")
#export DIR_PREFIX=$(dirname "$DIR_PREFIX")
#DIR_PREFIX=/media/MYPASSPORT2/Professional/CRL/06-BrainExtraction6_rad1

export MIALSRTK_DIR="${MIALSRTK_PREFIX}"
export MIALSRTK_UTILITIES=$MIALSRTK_DIR/Utilities
export MIALSRTK_APPLICATIONS=$MIALSRTK_DIR/Applications
export MIALSRTK_CRKIT=$MIALSRTK_DIR/CRKit
printf "MIALSRTK_DIR=${MIALSRTK_DIR} \n"

BTK_HOME=${MIALSRTK_PREFIX}/External/fbrain-build
export BTK_UTILITIES=$BTK_HOME/Utilities
export BTK_APPLICATIONS=$BTK_HOME/Applications

PATIENT_DIR="$(dirname "$0")"
echo "Scan list file : ${1}"
echo "Working directory : $PATIENT_DIR"

RESULTS=${PATIENT_DIR}/RECON
if [ ! -d "${RESULTS}" ]; then
  mkdir -p "${RESULTS}";
  echo "Folder ${RESULTS} created"
fi
echo "Reconstruction directory: $RESULTS"

SCANS="${1}"
echo "List of scans : $SCANS"

##Localization
templateName="T${GA}template"
BRAIN_LOC="${PATIENT_DIR}/${templateName}"

if [ ! -d "${BRAIN_LOC}" ]; then
  mkdir -p "${BRAIN_LOC}";
  echo "Folder ${BRAIN_LOC} created"
fi

template="${ATLAS_DIR}/${templateName}.nii"
templateMask="${ATLAS_DIR}/${templateName}_brain_mask.nii"
echo "Brain localization directory: $BRAIN_LOC"

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

##Count number of scans used for reconstruction
VOLS=0
while read -r line
do
	((VOLS = VOLS + 1))
done < "$SCANS"

echo 
echo "-----------------------------------------------"
echo
echo "Number of scans : ${VOLS}"
echo 
echo "-----------------------------------------------"
echo

##Iteration of motion estimation / reconstruction / brain mask refinement
for (( ITER=$START_ITER; ITER<=$MAX_ITER; ITER++ ))
do
	echo "Performing iteration # ${ITER}"

	cmdIntensity="$MIALSRTK_UTILITIES/mialtkIntensityStandardization"
	cmdIntensityNLM="$MIALSRTK_UTILITIES/mialtkIntensityStandardization"

	if [ "$ITER" -eq "1" ];
	then

		while read -r line
		do
			set -- $line
			stack=$1
			orientation=$2

			echo "Process stack $stack with $orientation orientation..."

		    #Reorient the image
			$MIALSRTK_UTILITIES/mialtkOrientImage -i $BRAIN_LOC/${stack}_Final.nii.gz -o $RESULTS/${stack}_Final_reo_iteration_${ITER}.nii.gz -O "$orientation"
			$MIALSRTK_UTILITIES/mialtkOrientImage -i $BRAIN_LOC/${stack}_Final_brain_mask${maskType}.nii.gz -o $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz -O "$orientation"

			#denoising on reoriented images
			weight="0.1"
			$BTK_APPLICATIONS/btkNLMDenoising -i "$RESULTS/${stack}_Final_reo_iteration_${ITER}.nii.gz" -o "$RESULTS/${stack}_Final_nlm_reo_iteration_${ITER}.nii.gz" -b $weight

			#Make slice intensities uniform in the stack
			$MIALSRTK_UTILITIES/mialtkCorrectSliceIntensity "$RESULTS/${stack}_Final_nlm_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_nlm_uni_reo_iteration_${ITER}.nii.gz"
			$MIALSRTK_UTILITIES/mialtkCorrectSliceIntensity "$RESULTS/${stack}_Final_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_uni_reo_iteration_${ITER}.nii.gz"

			#bias field correction slice by slice
			$MIALSRTK_UTILITIES/mialtkSliceBySliceN4BiasFieldCorrection "$RESULTS/${stack}_Final_nlm_uni_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_nlm_n4bias.nii.gz"
			$MIALSRTK_UTILITIES/mialtkSliceBySliceCorrectBiasField "$RESULTS/${stack}_Final_uni_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_nlm_n4bias.nii.gz" "$RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz"

			$MIALSRTK_UTILITIES/mialtkCorrectSliceIntensity "$RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			$MIALSRTK_UTILITIES/mialtkCorrectSliceIntensity "$RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz"

			#Intensity rescaling cmd preparation
			cmdIntensityNLM="$cmdIntensityNLM -i $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdIntensity="$cmdIntensity -i $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz"
		done < "$SCANS"

		echo "$cmdIntensity"

	else

		while read -r line
		do
			set -- $line
			stack=$1

			#Make slice intensities uniform in the stack
			$MIALSRTK_UTILITIES/mialtkCorrectSliceIntensity "$RESULTS/${stack}_Final_nlm_reo_iteration_1.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_nlm_uni_reo_iteration_${ITER}.nii.gz"
			$MIALSRTK_UTILITIES/mialtkCorrectSliceIntensity "$RESULTS/${stack}_Final_reo_iteration_1.nii.gz" "$RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz" "$RESULTS/${stack}_Final_uni_reo_iteration_${ITER}.nii.gz"

			cmdCorrectBiasField="$MIALSRTK_UTILITIES/mialtkCorrectBiasFieldWithMotionApplied -i $RESULTS/${stack}_Final_nlm_uni_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -o $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --input-bias-field $RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}_gbcorrfield.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --output-bias-field $RESULTS/${stack}_Final_nlm_n4bias_iteration_${ITER}.nii.gz" 
			cmdCorrectBiasField="$cmdCorrectBiasField -t $RESULTS/${stack}_Final_transform_${VOLS}V_${LAST_ITER}.txt"
			eval "$cmdCorrectBiasField"

			cmdCorrectBiasField="$MIALSRTK_UTILITIES/mialtkCorrectBiasFieldWithMotionApplied -i $RESULTS/${stack}_Final_uni_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField -o $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --input-bias-field $RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}_gbcorrfield.nii.gz"
			cmdCorrectBiasField="$cmdCorrectBiasField --output-bias-field $RESULTS/${stack}_Final_n4bias_iteration_${ITER}.nii.gz" 
			cmdCorrectBiasField="$cmdCorrectBiasField -t $RESULTS/${stack}_Final_transform_${VOLS}V_${LAST_ITER}.txt"
			eval "$cmdCorrectBiasField"

			cmdIntensityNLM="$cmdIntensityNLM -i $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			cmdIntensity="$cmdIntensity -i $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz -o $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}.nii.gz"
			
		done < "$SCANS"

	fi

	
	#Intensity rescaling
	eval "$cmdIntensityNLM"
	eval "$cmdIntensity"

	#histogram normalization - need to change the brain mask name expected according to the one used (full auto/localization and rigid extraction/localization only/manual)
	python ${PATIENT_DIR}/mialtkHistogramNormalizationNLM.py -i "${RESULTS}" -m "${RESULTS}" -t "${maskType}" -o "${RESULTS}" -I "${ITER}"
	python ${PATIENT_DIR}/mialtkHistogramNormalization.py -i "${RESULTS}" -m "${RESULTS}" -t "${maskType}" -o "${RESULTS}" -I "${ITER}"

	cmdIntensity="$MIALSRTK_UTILITIES/mialtkIntensityStandardization"
	cmdIntensityNLM="$MIALSRTK_UTILITIES/mialtkIntensityStandardization"
	while read -r line
	do
		set -- $line
		stack=$1
		#Intensity rescaling cmd preparation
		cmdIntensityNLM="$cmdIntensityNLM -i $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz -o $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdIntensity="$cmdIntensity -i $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz -o $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
	done < "$SCANS"

	#Intensity rescaling
	eval "$cmdIntensityNLM"
	eval "$cmdIntensity"

	echo "Initialize the super-resolution image using initial masks - Iteration ${ITER}..."

	cmdImageRECON="$MIALSRTK_APPLICATIONS/mialtkImageReconstruction --mask"
	cmdSuperResolution="$MIALSRTK_APPLICATIONS/mialtkTVSuperResolutionWithImplicitGradientDescent"
	#cmdRobustSuperResolution="$MIALSRTK_APPLICATIONS/mialtkRobustTVSuperResolutionWithGMM"

	#Preparation for (1) motion estimation and SDI reconstruction and (2) super-resolution reconstruction
	while read -r line
	do
		set -- $line
		stack=$1
		$MIALSRTK_UTILITIES/mialtkMaskImage -i "$RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz" -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz -o "$RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"

		cmdImageRECON="$cmdImageRECON -i $RESULTS/${stack}_Final_nlm_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdImageRECON="$cmdImageRECON -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz"
		cmdImageRECON="$cmdImageRECON -t $RESULTS/${stack}_Final_transform_${VOLS}V_${ITER}.txt"

		cmdSuperResolution="$cmdSuperResolution -i $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdSuperResolution="$cmdSuperResolution -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz"
		cmdSuperResolution="$cmdSuperResolution -t $RESULTS/${stack}_Final_transform_${VOLS}V_${ITER}.txt"
		#cmdRobustSuperResolution="$cmdRobustSuperResolution -i $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz -t $RESULTS/${stack}_Final_transform_${VOLS}V_${ITER}.txt"
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

	cmdSuperResolution="$cmdSuperResolution -o $RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}.nii.gz" 
	cmdSuperResolution="$cmdSuperResolution -r $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz" 
	cmdSuperResolution="$cmdSuperResolution --bregman-loop 1 --loop ${LOOPS} --iter 50 --step-scale 10 --gamma 10 --deltat ${DELTA_T}" 
	cmdSuperResolution="$cmdSuperResolution --lambda ${LAMBDA_TV} --inner-thresh 0.00001 --outer-thresh 0.000001"
	eval "$cmdSuperResolution"

	echo "Done"
	echo

	#up="1.0"
	#cmdRobustSuperResolution="$cmdRobustSuperResolution -o $RESULTS/RobustSRTV_${PATIENT}_${VOLS}V_NoNLM_bcorr_norm_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_it${ITER}_rad${RAD_DILATION}_up${up}.nii.gz -r $RESULTS/SDI_${PATIENT}_${VOLS}V_nlm_bcorr_norm_it${ITER}_rad${RAD_DILATION}.nii.gz --bregman-loop 1 --loop ${LOOPS} --iter 50 --step-scale 10 --gamma 10 --lambda ${LAMBDA_TV} --deltat ${DELTA_T} --inner-thresh 0.00001 --outer-thresh 0.000001 --use-robust --huber-mode 2 --upscaling-factor $up"
	#eval "$cmdRobustSuperResolution"

	NEXT_ITER=${ITER}
	((NEXT_ITER = NEXT_ITER + 1))

	echo "##########################################################################################################################"
	echo

	echo "Refine the mask of the HR image for next iteration (${NEXT_ITER})..."

	echo
	echo "##########################################################################################################################"
	echo

	#Preparation for brain mask refinement
	cmdRefineMasks="$MIALSRTK_UTILITIES/mialtkRefineHRMaskByIntersection --use-staple --radius-dilation ${RAD_DILATION}"
	while read -r line
	do
		set -- $line
		stack=$1
		cmdRefineMasks="$cmdRefineMasks -i $RESULTS/${stack}_Final_uni_bcorr_reo_iteration_${ITER}_histnorm.nii.gz"
		cmdRefineMasks="$cmdRefineMasks -m $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${ITER}.nii.gz"
		cmdRefineMasks="$cmdRefineMasks -t $RESULTS/${stack}_Final_transform_${VOLS}V_${ITER}.txt"
		cmdRefineMasks="$cmdRefineMasks -O $RESULTS/${stack}_Final_brain_mask${maskType}_reo_iteration_${NEXT_ITER}.nii.gz"
	done < "$SCANS"

	#Brain mask refinement

	cmdRefineMasks="$cmdRefineMasks -o $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_brain_mask.nii.gz -r $RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}.nii.gz"
	eval "$cmdRefineMasks"

	#Bias field refinement
	$MIALSRTK_UTILITIES/mialtkN4BiasFieldCorrection "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}.nii.gz" "$RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_brain_mask.nii.gz" "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${NEXT_ITER}_gbcorr.nii.gz" "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${NEXT_ITER}_gbcorrfield.nii.gz"

	#Brain masking of the reconstructed image

	$MIALSRTK_UTILITIES/mialtkMaskImage -i "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}.nii.gz" -m "$RESULTS/SDI_${PATIENT}_${VOLS}V_rad${RAD_DILATION}_it${ITER}_brain_mask.nii.gz" -o "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${ITER}_masked.nii.gz"

	echo
	echo "##########################################################################################################################"
	echo

	LAST_ITER="$ITER"

done

cp "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${LAST_ITER}.nii.gz" "$PATIENT_DIR/superresolution.nii.gz"
cp "$RESULTS/SRTV_${PATIENT}_${VOLS}V_lambda_${LAMBDA_TV}_deltat_${DELTA_T}_loops_${LOOPS}_rad${RAD_DILATION}_it${LAST_ITER}_masked.nii.gz" "$PATIENT_DIR/superresolution_masked.nii.gz"

END1=$(date +%s)

DIFF1=$(( $END1 - $START1 ))

echo "Done. It took $DIFF1 seconds for reconstruction, after $MAX_ITER refinement loops, using ${VOLS} volumes."
echo 
echo "-----------------------------------------------"
echo
