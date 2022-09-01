# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the postprocessing stage of the super-resolution
reconstruction pipeline."""

import numpy as np

from traits.api import *

import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.postprocess as postprocess
from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util
from nipype.interfaces.io import DataGrabber


def convert_ga(ga):
    ga = int(np.round(ga))
    if ga > 38:
        ga = 38
    elif ga < 21:
        ga = 21
    ga_str = str(ga) + 'exp' if ga > 35 else str(ga)
    return ga_str


def create_postproc_stage(
        p_ga,
        p_do_anat_orientation=False,
        p_do_reconstruct_labels=False,
        p_verbose=False,
        name="postproc_stage"
):
    """Create a SR preprocessing workflow
    Parameters
    ----------
        name : :str:
            name of workflow (default: preproc_stage)
        p_ga: :int:
            Subject's gestational age in weeks
        p_do_anat_orientation: :bool:
            Whether the alignement to template should be performed
        p_do_reconstruct_labels: :bool:
            Whether the reconstruction of LR labelmaps should be performed
        p_verbose: :bool:
            Whether verbosity is enabled.
    Inputs
    ------
        input_sdi:
            Input SDI image (filename)
        input_image:
            Input T2w image (filename)
        input_mask:
            Input mask image (filename)
        input_labelmap: (optional)
            Input labelmap image (filename)
    Outputs
    -------
        output_image :
            Postprocessed image (filename)
        output_mask :
            Postprocessed mask (filename)
        output_labelmap :
            Postprocessed labelmap (filename)
    """

    postproc_stage = pe.Workflow(name=name)

    # Set up a node to define all inputs for the postprocessing workflow

    input_fields = ['input_image', 'input_mask', 'input_sdi']
    output_fields = ['output_image', 'output_mask']

    if p_do_reconstruct_labels:
        input_fields += ['input_labelmap']
        output_fields += ['output_labelmap']

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=output_fields
        ),
        name='outputnode')

    srtkN4BiasFieldCorrection = pe.Node(
        interface=postprocess.MialsrtkN4BiasFieldCorrection(),
        name='srtkN4BiasFieldCorrection')
    srtkN4BiasFieldCorrection.inputs.verbose = p_verbose

    srtkMaskImage02 = pe.Node(
        interface=preprocess.MialsrtkMaskImage(),
        name='srtkMaskImage02')
    srtkMaskImage02.inputs.verbose = p_verbose

    if p_do_anat_orientation and p_ga is not None:

        ga_str = convert_ga(p_ga)

        atlas_grabber = pe.Node(
            interface=DataGrabber(outfields=['atlas', 'tissue']),
            name='atlas_grabber'
        )
        atlas_grabber.inputs.base_directory = '/sta'
        atlas_grabber.inputs.template = '*'
        atlas_grabber.inputs.raise_on_empty = True
        atlas_grabber.inputs.sort_filelist = True

        atlas_grabber.inputs.field_template = \
            dict(atlas='STA'+ga_str+'.nii.gz')

        resample_t2w_template = pe.Node(
            interface=preprocess.ResampleImage(),
            name='resample_t2w_template'
        )

        compute_alignment = pe.Node(
            interface=preprocess.ComputeAlignmentToReference(),
            name='compute_alignment'
        )

        align_image = pe.Node(
            interface=preprocess.ApplyAlignmentTransform(),
            name='align_image'
        )

        if p_do_reconstruct_labels:
            align_labelmap = pe.Node(
                interface=preprocess.ApplyAlignmentTransform(),
                name='align_labelmap'
            )
    if p_do_reconstruct_labels:
        mask_hr_label = pe.Node(
            interface=preprocess.MialsrtkMaskImage(),
            name='mask_hr_label'
        )

    postproc_stage.connect(inputnode, "input_image",
                           srtkMaskImage02, "in_file")
    postproc_stage.connect(inputnode, "input_mask",
                           srtkMaskImage02, "in_mask")

    postproc_stage.connect(srtkMaskImage02, "out_im_file",
                           srtkN4BiasFieldCorrection, "input_image")
    postproc_stage.connect(inputnode, "input_mask",
                           srtkN4BiasFieldCorrection, "input_mask")

    if p_do_reconstruct_labels:
        postproc_stage.connect(inputnode, "input_labelmap",
                               mask_hr_label, "in_file")
        postproc_stage.connect(inputnode, "input_mask",
                               mask_hr_label, "in_mask")

    if not p_do_anat_orientation:
        postproc_stage.connect(srtkN4BiasFieldCorrection, "output_image",
                               outputnode, "output_image")

        postproc_stage.connect(inputnode, "input_mask",
                               outputnode, "output_mask")

        if p_do_reconstruct_labels:
            postproc_stage.connect(mask_hr_label, "out_im_file",
                                   outputnode, "output_labelmap")

    else:
        postproc_stage.connect(srtkN4BiasFieldCorrection, "output_image",
                               resample_t2w_template, "input_reference")
        postproc_stage.connect(atlas_grabber, "atlas",
                               resample_t2w_template, "input_image")

        postproc_stage.connect(inputnode, "input_sdi",
                               compute_alignment, "input_image")
        postproc_stage.connect(resample_t2w_template, "output_image",
                               compute_alignment, "input_template")

        postproc_stage.connect(srtkN4BiasFieldCorrection, "output_image",
                               align_image, "input_image")
        postproc_stage.connect(resample_t2w_template, "output_image",
                               align_image, "input_template")

        postproc_stage.connect(inputnode, "input_mask",
                               align_image, "input_mask")

        postproc_stage.connect(compute_alignment, "output_transform",
                               align_image, "input_transform")

        postproc_stage.connect(align_image, "output_image",
                               outputnode, "output_image")
        postproc_stage.connect(align_image, "output_mask",
                               outputnode, "output_mask")

        if p_do_reconstruct_labels:
            postproc_stage.connect(srtkN4BiasFieldCorrection, "output_image",
                                   align_labelmap, "input_image")
            postproc_stage.connect(resample_t2w_template, "output_image",
                                   align_labelmap, "input_template")

            postproc_stage.connect(mask_hr_label, "out_im_file",
                                   align_labelmap, "input_mask")

            postproc_stage.connect(compute_alignment, "output_transform",
                                   align_labelmap, "input_transform")
            postproc_stage.connect(align_labelmap, "output_mask",
                                   outputnode, "output_labelmap")


    return postproc_stage
