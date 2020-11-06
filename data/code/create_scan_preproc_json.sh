#!/bin/sh
# 
# Usage:
# 		sh create_scan_preproc_json output.json source.nii.gz
# 
# Author: Sebastien Tourbier
# 
###################################################################

OUTPUT_JSON=$1

(
cat <<EOF
{
	"Description": "Preprocessed image used as input to the Super-Resolution algorithm",
    "Sources": "$2",
    "CustomMetaData": {
		"Reorient": true,
		"Intensity standardization":   true,
		"Bias Field Correction": true
	}
}

EOF
) > $OUTPUT_JSON
