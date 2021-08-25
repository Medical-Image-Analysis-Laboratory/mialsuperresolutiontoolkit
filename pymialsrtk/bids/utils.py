# Copyright © 2016-2021 Medical Image Analysis Laboratory,
# University Hospital Center and University of Lausanne (UNIL-CHUV),
# Switzerland, and contributors
#
#  This software is distributed under the open-source license Modified BSD.

"""This modules provides CMTK Utility functions to handle BIDS datasets."""

import os
import json

from nipype import __version__ as nipype_version
from pymialsrtk.info import __version__, __url__, DOCKER_HUB


def write_bids_derivative_description(bids_dir, deriv_dir, pipeline_name):
    """Write a dataset_description.json in each type of PyMIALSRTK derivatives.

    Parameters
    ----------
    bids_dir : string
        BIDS root directory
    deriv_dir : string
        Output/derivatives directory
    pipeline_name : string
        Type of derivatives (`['pymialsrtk', 'nipype']`)

    """
    bids_dir = os.path.abspath(bids_dir)
    deriv_dir = os.path.abspath(deriv_dir)

    if pipeline_name == "pymialsrtk":
        desc = {
            "Name": "PyMIALSRTK Outputs",
            "BIDSVersion": "1.4.0",
            "DatasetType": "derivatives",
            "GeneratedBy": {
                "Name": f'{pipeline_name}',
                "Version": __version__,
                "Container": {
                    "Type": "docker",
                    "Tag": "{}:{}".format(DOCKER_HUB, __version__),
                },
                "CodeURL": __url__,
            },
            "PipelineDescription": {"Name": "PyMIALSRTK Outputs ({})".format(__version__)},
            "HowToAcknowledge": "Please cite ... ",
        }
    elif pipeline_name == "nipype":
        desc = {
            "Name": "Nipype Outputs of PyMIALSRTK ({})".format(__version__),
            "BIDSVersion": "1.4.0",
            "DatasetType": "derivatives",
            "GeneratedBy": {
                "Name": pipeline_name,
                "Version": nipype_version,
                "Container": {
                    "Type": "docker",
                    "Tag": "{}:{}".format(DOCKER_HUB, __version__),
                },
                "CodeURL": __url__,
            },
            "PipelineDescription": {
                "Name": "Nipype Outputs of PyMIALSRTK ({})".format(__version__),
            },
            "HowToAcknowledge": "Please cite ... ",
        }

    # Keys deriving from source dataset
    orig_desc = {}
    fname = os.path.join(bids_dir, "dataset_description.json")
    if os.access(fname, os.R_OK):
        with open(fname, "r") as fobj:
            orig_desc = json.load(fobj)

    if "DatasetDOI" in orig_desc:
        desc["SourceDatasets"]: [
            {
                "DOI": orig_desc["DatasetDOI"],
                "URL": "https://doi.org/{}".format(orig_desc["DatasetDOI"]),
                "Version": "TODO: To be updated",
            }
        ]
    else:
        desc["SourceDatasets"]: [
            {
                "DOI": "TODO: To be updated",
                "URL": "TODO: To be updated",
                "Version": "TODO: To be updated",
            }
        ]

    desc[
        "License"
    ] = "TODO: To be updated (See https://creativecommons.org/about/cclicenses/)"

    # Save the dataset_description.json file
    output_filename = os.path.join(
        deriv_dir,
        f'{desc["GeneratedBy"]["Name"]}-{desc["GeneratedBy"]["Version"]}',
        "dataset_description.json"
    )
    print(f'\tSave {desc["GeneratedBy"]["Name"]} bids tool description to {output_filename}...')
    with open(output_filename, 'w+') as fobj:
        json.dump(desc, fobj, indent=4)
