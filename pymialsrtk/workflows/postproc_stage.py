# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the postprocessing stage of the super-resolution
reconstruction pipeline."""


from traits.api import *

from nipype.interfaces.base import (TraitedSpec, File, InputMultiPath,
                                    OutputMultiPath, BaseInterface,
                                    BaseInterfaceInputSpec)
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.postprocess as postprocess
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util


def create_postproc_stage(name="postproc_stage"):
    """Create a SR preprocessing workflow
    Parameters
    ----------
    ::
        name : name of workflow (default: preproc_stage)
    Inputs::
        inputnode.input_image : Input T2w image (filename)
        inputnode.input_mask : Input mask image (filenames)
    Outputs::
        outputnode.output_image : Postprocessed image (filename)
    Example
    -------
    >>> postproc_stage = create_preproc_stage(p_do_nlm_denoising=False)
    >>> postproc_stage.inputs.inputnode.input_image = 'sub-01_run-1_T2w.nii.gz'
    >>> postproc_stage.inputs.inputnode.input_mask = 'sub-01_run-1_T2w_mask.nii.gz'
    >>> postproc_stage.run() # doctest: +SKIP
    """

    postproc_stage = pe.Workflow(name=name)

    # Set up a node to define all inputs for the postprocessing workflow

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['input_image', 'input_mask']),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=['output_image']),
        name='outputnode')

    srtkN4BiasFieldCorrection = pe.Node(
        interface=postprocess.MialsrtkN4BiasFieldCorrection(),
        name='srtkN4BiasFieldCorrection')

    srtkMaskImage02 = pe.Node(
        interface=preprocess.MialsrtkMaskImage(),
        name='srtkMaskImage02')

    postproc_stage.connect(inputnode, "input_image",
                           srtkMaskImage02, "in_file")
    postproc_stage.connect(inputnode, "input_mask",
                           srtkMaskImage02, "in_mask")

    postproc_stage.connect(srtkMaskImage02, "out_im_file",
                           srtkN4BiasFieldCorrection, "input_image")
    postproc_stage.connect(inputnode, "input_mask",
                           srtkN4BiasFieldCorrection, "input_mask")

    postproc_stage.connect(srtkN4BiasFieldCorrection, "output_image",
                           outputnode, "output_image")

    return postproc_stage
