# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the various parts of workflow pipeline."""

from nipype.pipeline import engine as pe

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.reconstruction as reconstruction

# Get pymialsrtk version
from pymialsrtk.info import __version__
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
        p_do_nlm_denoising : :obj:`bool`
            Enable non-local means denoising (default: False)
        p_skip_svr : :obj:`bool`
            Skip slice-to-volume registration (default: False)
        p_sub_ses : :obj:`str`
            String containing subject-session information.
        name : :obj:`str`
            name of workflow (default: registration_stage)

    Inputs
    ------
        input_images :
            Input low-resolution T2w images (list of filenames)
        input_images_nlm :
            Input low-resolution denoised T2w images (list of filenames),
            Optional - only if p_do_nlm_denoising = True
        input_masks :
            Input mask images from the low-resolution T2w images
            (list of filenames)
        stacks_order :
            Order of stacks in the
            registration (list of integer)

    Outputs
    -------
        output_sdi :
            SDI image (filename)
        output_tranforms :
            Transfmation estimated parameters (list of filenames)

    # doctest: +SKIP
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
