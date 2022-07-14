# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the postprocessing stage of the super-resolution reconstruction pipeline."""

import os
import traceback
from glob import glob
import pathlib

from traits.api import *

from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.utils as utils

from nipype import config
from nipype import logging as nipype_logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util


def create_postproc_stage(bids_dir='', name="postproc_stage"):
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
    >>> postproc_stage = create_preproc_stage(bids_dir='/path/to/bids_dir', p_do_nlm_denoising=False)
    >>> postproc_stage.inputs.inputnode.input_image = 'sub-01_run-1_T2w.nii.gz'
    >>> postproc_stage.inputs.inputnode.input_mask = 'sub-01_run-1_T2w_mask.nii.gz'
    >>> postproc_stage.run() # doctest: +SKIP
    """

    postproc_stage = pe.Workflow(name=name)
    """
    Set up a node to define all inputs required for the preprocessing workflow
    """

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['input_image', 'input_mask']),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=['output_image']),
        name='outputnode')

    """
    """

    srtkN4BiasFieldCorrection = pe.Node(interface=postprocess.MialsrtkN4BiasFieldCorrection(),
                                     name='srtkN4BiasFieldCorrection')
    srtkN4BiasFieldCorrection.inputs.bids_dir = bids_dir

    srtkMaskImage02 = pe.Node(interface=preprocess.MialsrtkMaskImage(),
                           name='srtkMaskImage02')
    srtkMaskImage02.inputs.bids_dir = bids_dir


    postproc_stage.connect(inputnode, "input_image",
                           srtkMaskImage02, "in_file")
    postproc_stage.connect(inputnode, "input_mask",
                           srtkMaskImage02, "in_mask")

    postproc_stage.connect(srtkMaskImage02, "out_im_file",
                           srtkN4BiasFieldCorrection, "input_image")
    postproc_stage.connect(inputnode, "input_mask",
                           srtkN4BiasFieldCorrection, "input_mask")

    postproc_stage.connect(srtkN4BiasFieldCorrection, "output_image",
                           outputnode, "outputnode.output_image")

    # postproc_stage.connect(srtkN4BiasFieldCorrection, "output_field",
    #                        outputnode, "outputnode.output_field")
    return postproc_stage
