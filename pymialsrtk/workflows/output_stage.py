# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the management of the output of super-resolution
reconstruction pipeline."""

from traits.api import *
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.postprocess as postprocess
from nipype.interfaces.io import DataSink


def create_srr_output_stage(p_do_nlm_denoising=False,
                            p_do_reconstruct_labels=False,
                            p_skip_stacks_ordering=False,
                            name="srr_output_stage"):
    """Create a output management workflow for the
    super-resolution reconstruction pipeline.

    Parameters
    ----------
    p_do_nlm_denoising : :obj:`bool`
        Enable non-local means denoising (default: False)
    p_do_reconstruct_labels: :obj:`bool`
        Enable the reconstruction of labelmaps
    p_skip_stacks_ordering :  :obj:`bool`
        Skip stacks ordering (default: False)
        If disabled, `report_image` and `motion_tsv` are not generated
    name : :obj:`str`
        name of workflow (default: "srr_output_stage")

    Inputs
    ------
    sub_ses
        String containing subject-session information for output formatting
    sr_id
        ID of the current run
    stacks_order
        Order of stacks in the registration (list of integer)
    use_manual_masks
        Whether manual masks were used in the pipeline
    final_res_dir
        Output directory
    run_type
        Type of run (preprocessing/super resolution/ ...)
    input_masks
        Input mask images from the low-resolution T2w images
        (list of filenames)
    input_images
        Input low-resolution T2w images (list of filenames)
    input_transforms
        Transforms obtained after SVR
    input_sdi
        Interpolated high resolution volume, obtained after
        slice-to-volume registration (SVR)
    input_sr
        High resolution volume, obtained after the super-
        resolution (SR) reconstruction from the SDI volume.
    input_hr_mask
        Brain mask from the high-resolution reconstructed
        volume.
    input_json_path
        Path to the JSON file describing the parameters
        used in the SR reconstruction.
    input_sr_png
        PNG image summarizing the SR reconstruction.
    report_image
        Report image obtained from the StacksOrdering module
        Optional - only if p_skip_stacks_ordering = False
    motion_tsv
        Motion index obtained from the StacksOrdering module
        Optional - only if p_skip_stacks_ordering = False
    input_images_nlm
        Input low-resolution denoised T2w images
        Optional - only if p_do_nlm_denoising = True

    """

    srr_output_stage = pe.Workflow(name=name)
    # Set up a node to define all inputs required for the srr output workflow
    input_fields = ["sub_ses", "sr_id", "stacks_order", "use_manual_masks",
                    "final_res_dir", "run_type"]
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

    # Datasinker
    finalFilenamesGeneration = pe.Node(
        interface=postprocess.FilenamesGeneration(),
        name='filenames_gen')

    datasink = pe.Node(interface=DataSink(), name='data_sinker')

    srr_output_stage.connect(inputnode, "sub_ses", finalFilenamesGeneration,
                             "sub_ses")
    srr_output_stage.connect(inputnode, "sr_id", finalFilenamesGeneration,
                             "sr_id")
    srr_output_stage.connect(inputnode, "stacks_order",
                             finalFilenamesGeneration, "stacks_order")
    srr_output_stage.connect(inputnode, "use_manual_masks",
                             finalFilenamesGeneration, "use_manual_masks")
    srr_output_stage.connect(inputnode, "run_type",
                             finalFilenamesGeneration, 'run_type')

    srr_output_stage.connect(finalFilenamesGeneration, "substitutions",
                             datasink, "substitutions")

    srr_output_stage.connect(inputnode, "final_res_dir", datasink,
                             'base_directory')

    srr_output_stage.connect(inputnode, "input_masks",
                             datasink, 'anat.@LRmasks')
    srr_output_stage.connect(inputnode, "input_images",
                             datasink, 'anat.@LRsPreproc')
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

    if p_do_nlm_denoising:
        srr_output_stage.connect(inputnode, "input_images_nlm",
                                 datasink, 'anat.@LRsDenoised')

    if p_do_reconstruct_labels:
        srr_output_stage.connect(inputnode, "input_labelmap",
                                 datasink, 'anat.@SRlabelmap')

    if not p_skip_stacks_ordering:
        srr_output_stage.connect(inputnode, "report_image",
                                 datasink, 'figures.@stackOrderingQC')
        srr_output_stage.connect(inputnode, "motion_tsv",
                                 datasink, 'anat.@motionTSV')

    return srr_output_stage


def create_preproc_output_stage(p_do_nlm_denoising=False,
                                p_skip_stacks_ordering=False,
                                p_do_registration=False,
                                name="preproc_output_stage"):
    """Create an output management workflow for
    the preprocessing only pipeline.

    Parameters
    ----------
    p_do_nlm_denoising : :obj:`bool`
        Enable non-local means denoising (default: False)
    p_skip_stacks_ordering :  :obj:`bool`
        Skip stacks ordering (default: False)
        If disabled, `report_image` and `motion_tsv` are not generated
    p_do_registration : :obj:`bool`
        Whether registration is performed in the preprocessing pipeline
    name : :obj:`str`
        name of workflow (default: "preproc_output_stage")

    Inputs
    ------
    sub_ses
        String containing subject-session information for output formatting
    sr_id
        ID of the current run
    stacks_order
        Order of stacks in the registration (list of integer)
    use_manual_masks
        Whether manual masks were used in the pipeline
    final_res_dir
        Output directory
    run_type
        Type of run (preprocessing/super resolution/ ...)
    input_masks
        Input mask images from the low-resolution T2w images
        (list of filenames)
    input_images
        Input low-resolution T2w images (list of filenames)
    input_sdi
        Interpolated high resolution volume, obtained after
        slice-to-volume registration (SVR)
        Optional - only if p_do_registration = True
    input_transforms
        Transforms obtained after SVR
        Optional - only if p_do_registration = True
    report_image
        Report image obtained from the StacksOrdering module
        Optional - only if p_skip_stacks_ordering = False
    motion_tsv
        Motion index obtained from the StacksOrdering module
        Optional - only if p_skip_stacks_ordering = False
    input_images_nlm
        Input low-resolution denoised T2w images (list of filenames),
        Optional - only if p_do_nlm_denoising = True
    """

    prepro_output_stage = pe.Workflow(name=name)
    # Set up a node to define all inputs required for the srr output workflow
    input_fields = ["sub_ses", "sr_id", "stacks_order",
                    "use_manual_masks", "final_res_dir",
                    "run_type"
                    ]
    input_fields += ["input_masks", "input_images"]
    if p_do_registration:
        input_fields += ["input_sdi", "input_transforms"]
    if not p_skip_stacks_ordering:
        input_fields += ['report_image', 'motion_tsv']
    if p_do_nlm_denoising:
        input_fields += ['input_images_nlm']

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    # Datasinker
    finalFilenamesGeneration = pe.Node(
        interface=postprocess.FilenamesGeneration(),
        name='filenames_gen')

    datasink = pe.Node(interface=DataSink(), name='data_sinker')

    prepro_output_stage.connect(inputnode, "sub_ses",
                                finalFilenamesGeneration, "sub_ses")
    prepro_output_stage.connect(inputnode, "sr_id",
                                finalFilenamesGeneration, "sr_id")
    prepro_output_stage.connect(inputnode, "stacks_order",
                                finalFilenamesGeneration, "stacks_order")
    prepro_output_stage.connect(inputnode, "use_manual_masks",
                                finalFilenamesGeneration, "use_manual_masks")
    prepro_output_stage.connect(inputnode, "run_type",
                                finalFilenamesGeneration, "run_type")

    prepro_output_stage.connect(finalFilenamesGeneration, "substitutions",
                                datasink, "substitutions")

    prepro_output_stage.connect(inputnode, "final_res_dir",
                                datasink, 'base_directory')

    if not p_skip_stacks_ordering:
        prepro_output_stage.connect(inputnode, "report_image",
                                    datasink, 'figures.@stackOrderingQC')
        prepro_output_stage.connect(inputnode, "motion_tsv",
                                    datasink, 'anat.@motionTSV')
    prepro_output_stage.connect(inputnode, "input_masks",
                                datasink, 'anat.@LRmasks')
    prepro_output_stage.connect(inputnode, "input_images",
                                datasink, 'anat.@LRsPreproc')
    if p_do_registration:
        prepro_output_stage.connect(inputnode, "input_transforms",
                                    datasink, 'xfm.@transforms')
        prepro_output_stage.connect(inputnode, "input_sdi",
                                    datasink, 'anat.@SDI')
    if p_do_nlm_denoising:
        prepro_output_stage.connect(inputnode, "input_images_nlm",
                                    datasink, 'anat.@LRsDenoised')

    return prepro_output_stage
