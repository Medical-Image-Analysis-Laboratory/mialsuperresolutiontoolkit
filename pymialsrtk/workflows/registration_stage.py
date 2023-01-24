# Copyright © 2016-2023 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the registration stage of the super-resolution reconstruction pipeline."""

from nipype.pipeline import engine as pe

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.reconstruction as reconstruction

# Get pymialsrtk version
from nipype.interfaces import utility as util


def create_registration_stage(
    p_do_nlm_denoising=False,
    p_skip_svr=False,
    p_sub_ses="",
    p_verbose=False,
    name="registration_stage",
):
    """Create a a registration workflow, used as an optional stage in the preprocessing only pipeline.

    Parameters
    ----------
    p_do_nlm_denoising : boolean
        Enable non-local means denoising
        (default: `False`)
    p_skip_svr : boolean
        Skip slice-to-volume registration
        (default: `False`)
    p_sub_ses : string
        String containing subject-session information.
    name : string
        name of workflow
        (default: "registration_stage")

    Inputs
    ------
    input_images : list of items which are a pathlike object or string representing a file
        Input low-resolution T2w images
    input_images_nlm : list of items which are a pathlike object or string representing a file
        Input low-resolution denoised T2w images,
        Optional - only if `p_do_nlm_denoising = True`
    input_masks : list of items which are a pathlike object or string representing a file
        Input mask images from the low-resolution T2w images
    stacks_order : list of integer
        Order of stacks in the registration

    Outputs
    -------
    output_sdi : pathlike object or string representing a file
        SDI image
    output_tranforms : list of items which are a pathlike object or string representing a file
        Estimated transformation parameters

    Example
    -------
    >>> from pymialsrtk.pipelines.workflows import registration_stage as reg
    >>> registration_stage = reg.create_registration_stage(
            p_sub_ses=p_sub_ses,
        )
    >>> registration_stage.inputs.input_images = [
            'sub-01_run-1_T2w.nii.gz',
            'sub-01_run-2_T2w.nii.gz'
        ]
    >>> registration_stage.inputs.input_masks = [
            'sub-01_run-1_T2w.nii_mask.gz',
            'sub-01_run-2_T2w.nii_mask.gz'
        ]
    >>> registration_stage.inputs.stacks_order = [2,1]
    >>> registration_stage.run()  # doctest: +SKIP

    """

    registration_stage = pe.Workflow(name=name)
    # Set up a node to define all inputs required for the preproc workflow
    input_fields = ["input_images", "input_masks", "stacks_order"]

    if p_do_nlm_denoising:
        input_fields += ["input_images_nlm"]

    # Input node with the input fields specified above
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=input_fields), name="inputnode"
    )

    # Output node with the interpolated HR image + transforms from registration
    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=["output_sdi", "output_transforms"]
        ),
        name="outputnode",
    )

    srtkImageReconstruction = pe.Node(
        interface=reconstruction.MialsrtkImageReconstruction(
            sub_ses=p_sub_ses, skip_svr=p_skip_svr, verbose=p_verbose
        ),
        name="srtkImageReconstruction",
    )

    if p_do_nlm_denoising:
        sdiComputation = pe.Node(
            interface=reconstruction.MialsrtkSDIComputation(
                sub_ses=p_sub_ses, verbose=p_verbose
            ),
            name="sdiComputation",
        )

    registration_stage.connect(
        inputnode, "input_masks", srtkImageReconstruction, "input_masks"
    )
    registration_stage.connect(
        inputnode, "stacks_order", srtkImageReconstruction, "stacks_order"
    )

    if p_do_nlm_denoising:
        registration_stage.connect(
            inputnode,
            "input_images_nlm",
            srtkImageReconstruction,
            "input_images",
        )

        registration_stage.connect(
            inputnode, "stacks_order", sdiComputation, "stacks_order"
        )
        registration_stage.connect(
            inputnode, "input_images_nlm", sdiComputation, "input_images"
        )
        registration_stage.connect(
            inputnode, "input_masks", sdiComputation, "input_masks"
        )
        registration_stage.connect(
            srtkImageReconstruction,
            "output_transforms",
            sdiComputation,
            "input_transforms",
        )
        registration_stage.connect(
            srtkImageReconstruction,
            "output_sdi",
            sdiComputation,
            "input_reference",
        )

    else:
        registration_stage.connect(
            inputnode, "input_images", srtkImageReconstruction, "input_images"
        )

    if p_do_nlm_denoising:
        registration_stage.connect(
            sdiComputation, "output_sdi", outputnode, "output_sdi"
        )
    else:
        registration_stage.connect(
            srtkImageReconstruction, "output_sdi", outputnode, "output_sdi"
        )

    registration_stage.connect(
        srtkImageReconstruction,
        "output_transforms",
        outputnode,
        "output_transforms",
    )

    return registration_stage
