#!/bin/sh
# 
# Usage:
# 		sh create_dataset_description_json output.json v1.1.0
# 
# Author: Sebastien Tourbier
# 
###################################################################

OUTPUT_JSON=$1

(
cat <<EOF
{
    "PipelineDescription": {
		"Name": "MIAL Super-Resolution ToolKit",
		"Version":   "$2",
		"CodeURL":   "https://github.com/sebastientourbier/mialsuperresolutiontoolkit"
	}
}

EOF
) > $OUTPUT_JSON
