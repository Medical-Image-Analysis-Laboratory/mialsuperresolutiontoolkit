# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the management of the output of super-resolution reconstruction pipeline."""

import os
import traceback
from glob import glob
import pathlib

from traits.api import *

from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces import utility as util

from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.utils as utils

from nipype import config
from nipype import logging as nipype_logging

from nipype.interfaces.io import DataSink


def create_srr_output_stage(p_do_nlm_denoising=False,
                            p_do_reconstruct_labels=False,
                            p_skip_stacks_ordering=False,
                            name="srr_output_stage"):
    """Create a output management workflow
    for srr pipeline
    Parameters
    ----------
    ::
        name : name of workflow (default: preproc_stage)
    Inputs::

    Outputs::

    Example
    -------
    >>>
    """


    srr_output_stage = pe.Workflow(name=name)
    """
    Set up a node to define all inputs required for the srr output workflow
    """
    input_fields = ["sub_ses", "sr_id", "stacks_order", "use_manual_masks", "final_res_dir"]

    input_fields += ["input_masks", "input_images", "input_transforms"]

    input_fields += ["input_sdi", "input_sr", "input_hr_mask"]

    input_fields += ["input_json_path", "input_sr_png"]

    if not p_skip_stacks_ordering:
        input_fields += ['report_image', 'motion_tsv']

    if p_do_nlm_denoising:
        input_fields += ['input_images_nlm']

    if p_do_reconstruct_labels:
        input_fields += ['input_labelmap']

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')


    """
    """

    # Datasinker
    finalFilenamesGeneration = pe.Node(
        interface=postprocess.FilenamesGeneration(),
        name='filenames_gen')

    datasink = pe.Node(interface=DataSink(), name='data_sinker')

    srr_output_stage.connect(inputnode, "sub_ses", finalFilenamesGeneration, "sub_ses")
    srr_output_stage.connect(inputnode, "sr_id", finalFilenamesGeneration, "sr_id")
    srr_output_stage.connect(inputnode, "stacks_order", finalFilenamesGeneration, "stacks_order")
    srr_output_stage.connect(inputnode, "use_manual_masks", finalFilenamesGeneration, "use_manual_masks")

    srr_output_stage.connect(finalFilenamesGeneration, "substitutions",
                    datasink, "substitutions")

    srr_output_stage.connect(inputnode, "final_res_dir", datasink, 'base_directory')
    srr_output_stage.connect(inputnode, "input_masks",
                             datasink, 'anat.@LRmasks')
    srr_output_stage.connect(inputnode, "input_images",
                             datasink, 'anat.@LRsPreproc')
    if p_do_nlm_denoising:
        srr_output_stage.connect(inputnode, "input_images_nlm",
                                 datasink, 'anat.@LRsDenoised')
    srr_output_stage.connect(inputnode, "input_transforms",
                             datasink, 'xfm.@transforms')

    srr_output_stage.connect(inputnode, "input_sdi",
                             datasink, 'anat.@SDI')
    srr_output_stage.connect(inputnode, "input_sr",
                             datasink, 'anat.@SR')
    srr_output_stage.connect(inputnode, "input_json_path",
                             datasink, 'anat.@SRjson')
    srr_output_stage.connect(inputnode, "input_sr_png",
                             datasink, 'figures.@SRpng')
    srr_output_stage.connect(inputnode, "input_hr_mask",
                             datasink, 'anat.@SRmask')

    if p_do_reconstruct_labels:
        srr_output_stage.connect(inputnode, "input_labelmap",
                        datasink, 'anat.@SRlabelmap')

    if not p_skip_stacks_ordering:
        srr_output_stage.connect(inputnode, "report_image",
                                 datasink, 'figures.@stackOrderingQC')
        srr_output_stage.connect(inputnode, "motion_tsv",
                                 datasink, 'anat.@motionTSV')

    return srr_output_stage
