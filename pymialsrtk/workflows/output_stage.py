# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#  This software is distributed under the open-source license Modified BSD.

"""Workflow for the management of the output of super-resolution
reconstruction pipeline."""

from pymialsrtk import interfaces
from traits.api import *
from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.postprocess as postprocess
from nipype.interfaces.io import DataSink


def create_srr_output_stage(
    p_sub_ses,
    p_sr_id,
    p_run_type,
    p_keep_all_outputs=False,
    p_use_manual_masks=False,
    p_do_nlm_denoising=False,
    p_do_reconstruct_labels=False,
    p_do_srr_assessment=False,
    p_skip_stacks_ordering=False,
    p_do_multi_parameters=False,
    p_subject=None,
    p_session=None,
    p_stacks=None,
    p_output_dir=None,
    p_run_start_time=None,
    p_run_elapsed_time=None,
    p_skip_svr=None,
    p_do_anat_orientation=None,
    p_do_refine_hr_mask=None,
    p_masks_derivatives_dir=None,
    p_openmp_number_of_cores=None,
    p_nipype_number_of_cores=None,
    name="srr_output_stage",
):
    """Create a output management workflow for the
    super-resolution reconstruction pipeline.

    Parameters
    ----------
    p_sub_ses :
        String containing subject-session information for output formatting
    p_sr_id :
        ID of the current run
    p_run_type :
        Type of run (preprocessing/super resolution/ ...)
    p_keep_all_outputs :
        Whether intermediate outputs must be issues
    p_use_manual_masks :
        Whether manual masks were used in the pipeline
    p_do_nlm_denoising : :obj:`bool`
        Enable non-local means denoising (default: False)
    p_do_reconstruct_labels: :obj:`bool`
        Enable the reconstruction of labelmaps
    p_do_srr_assessment: :obj:`bool
        Enables output of srr assessment stage
    p_skip_stacks_ordering :  :obj:`bool`
        Skip stacks ordering (default: False)
        If disabled, `report_image` and `motion_tsv` are not generated
    p_do_multi_parameters :  :obj:`bool`
        Whether recon_stage was performed in a multi-TV mode
    name : :obj:`str`
        name of workflow (default: "srr_output_stage")

    Inputs
    ------
    stacks_order
        Order of stacks in the registration (list of integer)
    use_manual_masks
        Whether manual masks were used in the pipeline
    final_res_dir
        Output directory

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
    input_fields = ["stacks_order", "final_res_dir"]
    input_fields += ["input_masks", "input_images", "input_transforms"]
    input_fields += ["input_sdi", "input_sr", "input_hr_mask"]
    input_fields += ["input_json_path", "input_sr_png"]

    input_fields += ["input_sr_heatmap"]

    if not p_skip_stacks_ordering:
        input_fields += ["report_image", "motion_tsv"]

    if p_do_nlm_denoising:
        input_fields += ["input_images_nlm"]

    if p_do_srr_assessment:
        input_fields += ["input_metrics", "input_metrics_labels"]

    if p_do_reconstruct_labels:
        input_fields += ["input_labelmap"]

    if p_do_multi_parameters:
        input_fields += ["input_TV_params"]

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=input_fields), name="inputnode"
    )
    if not p_do_multi_parameters:
        # Report generation
        reportGenerator = pe.Node(
            interface=postprocess.ReportGeneration(
                subject=p_subject,
                session=p_session if p_session is not None else "",
                stacks=[] if p_stacks is None else p_stacks,
                sr_id=p_sr_id,
                run_type=p_run_type,
                output_dir=p_output_dir,
                run_start_time=0.0,  # p_run_start_time,
                run_elapsed_time=0.0,  # p_run_elapsed_time,
                skip_svr=p_skip_svr,
                do_nlm_denoising=p_do_nlm_denoising,
                do_refine_hr_mask=p_do_refine_hr_mask,
                do_reconstruct_labels=p_do_reconstruct_labels,
                do_anat_orientation=p_do_anat_orientation,
                do_multi_parameters=p_do_multi_parameters,
                do_srr_assessment=p_do_srr_assessment,
                skip_stacks_ordering=p_skip_stacks_ordering,
                masks_derivatives_dir=p_masks_derivatives_dir
                if p_masks_derivatives_dir is not None
                else "",
                openmp_number_of_cores=p_openmp_number_of_cores,
                nipype_number_of_cores=p_nipype_number_of_cores,
            ),
            name="report_gen",
        )
    # Datasinker
    finalFilenamesGeneration = pe.Node(
        interface=postprocess.FilenamesGeneration(
            sub_ses=p_sub_ses,
            sr_id=p_sr_id,
            run_type=p_run_type,
            use_manual_masks=p_use_manual_masks,
            multi_parameters=p_do_multi_parameters,
        ),
        name="filenames_gen",
    )

    datasink = pe.Node(interface=DataSink(), name="data_sinker")

    srr_output_stage.connect(
        inputnode, "stacks_order", finalFilenamesGeneration, "stacks_order"
    )

    if p_do_multi_parameters:
        srr_output_stage.connect(
            inputnode, "input_TV_params", finalFilenamesGeneration, "TV_params"
        )

    srr_output_stage.connect(
        finalFilenamesGeneration, "substitutions", datasink, "substitutions"
    )

    srr_output_stage.connect(
        inputnode, "final_res_dir", datasink, "base_directory"
    )

    if p_keep_all_outputs:
        srr_output_stage.connect(
            inputnode, "input_masks", datasink, "anat.@LRmasks"
        )
        srr_output_stage.connect(
            inputnode, "input_images", datasink, "anat.@LRsPreproc"
        )
        srr_output_stage.connect(
            inputnode, "input_transforms", datasink, "xfm.@transforms"
        )
        srr_output_stage.connect(inputnode, "input_sdi", datasink, "anat.@SDI")
        if p_do_nlm_denoising:
            srr_output_stage.connect(
                inputnode, "input_images_nlm", datasink, "anat.@LRsDenoised"
            )

    srr_output_stage.connect(inputnode, "input_sr", datasink, "anat.@SR")
    srr_output_stage.connect(
        inputnode, "input_json_path", datasink, "anat.@SRjson"
    )
    srr_output_stage.connect(
        inputnode, "input_sr_png", datasink, "figures.@SRpng"
    )
    srr_output_stage.connect(
        inputnode, "input_hr_mask", datasink, "anat.@SRmask"
    )

    srr_output_stage.connect(
        inputnode, "input_sr_heatmap", datasink, "anat.@SRheatmap"
    )

    if p_do_srr_assessment:
        srr_output_stage.connect(
            inputnode, "input_metrics", datasink, "anat.@SRmetrics"
        )
        srr_output_stage.connect(
            inputnode,
            "input_metrics_labels",
            datasink,
            "anat.@SRmetricsLabels",
        )

    if p_do_reconstruct_labels:
        srr_output_stage.connect(
            inputnode, "input_labelmap", datasink, "anat.@SRlabelmap"
        )

    if not p_skip_stacks_ordering:
        srr_output_stage.connect(
            inputnode, "report_image", datasink, "figures.@stackOrderingQC"
        )
        if p_keep_all_outputs:
            srr_output_stage.connect(
                inputnode, "motion_tsv", datasink, "anat.@motionTSV"
            )
    if not p_do_multi_parameters:
        srr_output_stage.connect(
            inputnode, "input_json_path", reportGenerator, "input_json_path"
        )
        srr_output_stage.connect(
            inputnode, "stacks_order", reportGenerator, "stacks_order"
        )
        srr_output_stage.connect(
            inputnode, "input_sr", reportGenerator, "input_sr"
        )
        srr_output_stage.connect(
            finalFilenamesGeneration,
            "substitutions",
            reportGenerator,
            "substitutions",
        )
        srr_output_stage.connect(
            reportGenerator, "report_html", datasink, "report"
        )
    return srr_output_stage


def create_preproc_output_stage(
    p_sub_ses,
    p_sr_id,
    p_run_type,
    p_use_manual_masks,
    p_do_nlm_denoising=False,
    p_skip_stacks_ordering=False,
    p_do_registration=False,
    name="preproc_output_stage",
):
    """Create an output management workflow for
    the preprocessing only pipeline.

    Parameters
    ----------
    p_sub_ses :
        String containing subject-session information for output formatting
    p_sr_id :
        ID of the current run
    p_run_type :
        Type of run (preprocessing/super resolution/ ...)
    p_use_manual_masks :
        Whether manual masks were used in the pipeline
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

    preproc_output_stage = pe.Workflow(name=name)
    # Set up a node to define all inputs required for the srr output workflow
    input_fields = ["stacks_order", "final_res_dir"]
    input_fields += ["input_masks", "input_images"]
    if p_do_registration:
        input_fields += ["input_sdi", "input_transforms"]
    if not p_skip_stacks_ordering:
        input_fields += ["report_image", "motion_tsv"]
    if p_do_nlm_denoising:
        input_fields += ["input_images_nlm"]

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=input_fields), name="inputnode"
    )

    # Datasinker
    finalFilenamesGeneration = pe.Node(
        interface=postprocess.FilenamesGeneration(
            sub_ses=p_sub_ses,
            sr_id=p_sr_id,
            run_type=p_run_type,
            use_manual_masks=p_use_manual_masks,
            multi_parameters=False,
        ),
        name="filenames_gen",
    )

    datasink = pe.Node(interface=DataSink(), name="data_sinker")

    preproc_output_stage.connect(
        inputnode, "stacks_order", finalFilenamesGeneration, "stacks_order"
    )
    preproc_output_stage.connect(
        finalFilenamesGeneration, "substitutions", datasink, "substitutions"
    )

    preproc_output_stage.connect(
        inputnode, "final_res_dir", datasink, "base_directory"
    )

    if not p_skip_stacks_ordering:
        preproc_output_stage.connect(
            inputnode, "report_image", datasink, "figures.@stackOrderingQC"
        )
        preproc_output_stage.connect(
            inputnode, "motion_tsv", datasink, "anat.@motionTSV"
        )
    preproc_output_stage.connect(
        inputnode, "input_masks", datasink, "anat.@LRmasks"
    )
    preproc_output_stage.connect(
        inputnode, "input_images", datasink, "anat.@LRsPreproc"
    )
    if p_do_registration:
        preproc_output_stage.connect(
            inputnode, "input_transforms", datasink, "xfm.@transforms"
        )
        preproc_output_stage.connect(
            inputnode, "input_sdi", datasink, "anat.@SDI"
        )
    if p_do_nlm_denoising:
        preproc_output_stage.connect(
            inputnode, "input_images_nlm", datasink, "anat.@LRsDenoised"
        )

    return preproc_output_stage
