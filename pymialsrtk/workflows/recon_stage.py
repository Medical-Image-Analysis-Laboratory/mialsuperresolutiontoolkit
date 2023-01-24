# Copyright © 2016-2023 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
# This software is distributed under the open-source license Modified BSD.

"""Module for the reconstruction stage of the super-resolution reconstruction pipeline."""

from traits.api import *

from nipype.interfaces import utility as util
from nipype.pipeline import engine as pe

import pymialsrtk.workflows.recon_labelmap_stage as recon_labelmap_stage

import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.utils as utils


def create_recon_stage(
    p_paramTV,
    p_use_manual_masks,
    p_do_multi_parameters=False,
    p_do_nlm_denoising=False,
    p_do_reconstruct_labels=False,
    p_do_refine_hr_mask=False,
    p_skip_svr=False,
    p_sub_ses="",
    p_verbose=False,
    name="recon_stage",
):
    """Create a super-resolution reconstruction workflow.

    Parameters
    ----------
    p_paramTV : :dict:
        Dictionary of TV parameters
    p_use_manual_masks : :bool:
        Whether masks were done manually.
    p_do_nlm_denoising : :bool:
        Whether to proceed to non-local mean denoising.
        (default: `False`)
    p_do_multi_parameters : :bool:
        Perform super-resolution reconstruction with
        a set of multiple parameters.
        (default: `False`)
    p_do_reconstruct_labels : :bool:
        Whether we are also reconstruction label maps.
        (default: `False`)
    p_do_refine_hr_mask : :bool:
        Whether to do high-resolution mask refinement.
        (default: `False`)
    p_skip_svr : :bool:
        Whether slice-to-volume registration (SVR) should
        be skipped.
        (default: `False`)
    p_sub_ses : :str:
        String describing subject-session information
        (default: '')
    p_verbose : :bool:
        Whether verbosity should be enabled
        (default: `False`)
    name : :str:
        Name of workflow
        (default: "recon_stage")

    Inputs
    ----------
    input_images : :list: of paths
        Input T2w images
    input_images_nlm : :list: of paths
        Input T2w images,
        required if `p_do_nlm_denoising=True`
    input_masks : list of paths
        Input mask images
    stacks_order : list of :int:
        Order of stacks in the reconstruction

    Outputs
    ----------
    output_sr : path
        SR reconstructed image
    output_sdi : path
        SDI image
    output_hr_mask : path
        SRR mask
    output_tranforms : :list: of paths
        Estimated transformation parameters
    outputnode.output_json_path : path
        Path to the json sidecar of the SR reconstruction
    outputnode.output_sr_png : path
        Path to the PNG of the SR reconstruction
    outputnode.output_TV_parameters : :dict:
        Parameters used for TV reconstruction

    Example
    -------
    >>> from pymialsrtk.pipelines.workflows import recon_stage as rec
    >>> recon_stage = rec.create_preproc_stage(
            p_paramTV,
            p_use_manual_masks,
            p_do_nlm_denoising=False)
    >>> recon_stage.inputs.inputnode.input_images =
            ['sub-01_run-1_T2w.nii.gz', 'sub-01_run-2_T2w.nii.gz']
    >>> recon_stage.inputs.inputnode.input_masks =
            ['sub-01_run-1_T2w_mask.nii.gz', 'sub-01_run-2_T2w_mask.nii.gz']
    >>> recon_stage.inputs.stacks_order = [2,1]
    >>> recon_stage.run()  # doctest: +SKIP
    """

    recon_stage = pe.Workflow(name=name)

    # Set up a node to define all inputs required for the
    # preprocessing workflow
    input_fields = ["input_images", "input_masks", "stacks_order"]

    if p_do_nlm_denoising:
        input_fields += ["input_images_nlm"]

    if p_do_reconstruct_labels:
        input_fields += ["input_labels"]

    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=input_fields), name="inputnode"
    )

    output_fields = [
        "output_sr",
        "output_sdi",
        "output_hr_mask",
        "output_transforms",
        "output_json_path",
        "output_sr_png",
        "output_TV_parameters",
    ]

    if p_do_reconstruct_labels:
        output_fields += ["output_labelmap"]
    if p_do_multi_parameters:
        output_fields += ["output_TV_params"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode",
    )

    # Setting default TV parameters if not defined
    deltatTV = (
        p_paramTV["deltatTV"] if "deltatTV" in p_paramTV.keys() else 0.01
    )
    lambdaTV = (
        p_paramTV["lambdaTV"] if "lambdaTV" in p_paramTV.keys() else 0.75
    )

    num_iterations = (
        p_paramTV["num_iterations"]
        if "num_iterations" in p_paramTV.keys()
        else 50
    )
    num_primal_dual_loops = (
        p_paramTV["num_primal_dual_loops"]
        if "num_primal_dual_loops" in p_paramTV.keys()
        else 10
    )
    num_bregman_loops = (
        p_paramTV["num_bregman_loops"]
        if "num_bregman_loops" in p_paramTV.keys()
        else 3
    )
    step_scale = (
        p_paramTV["step_scale"] if "step_scale" in p_paramTV.keys() else 1
    )
    gamma = p_paramTV["gamma"] if "gamma" in p_paramTV.keys() else 1

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

    srtkTVSuperResolution = pe.Node(
        interface=reconstruction.MialsrtkTVSuperResolution(
            sub_ses=p_sub_ses,
            use_manual_masks=p_use_manual_masks,
            verbose=p_verbose,
        ),
        name="srtkTVSuperResolution",
    )

    if p_do_multi_parameters:
        deltatTV = [deltatTV] if not isinstance(deltatTV, list) else deltatTV
        lambdaTV = [lambdaTV] if not isinstance(lambdaTV, list) else lambdaTV
        num_iterations = (
            [num_iterations]
            if not isinstance(num_iterations, list)
            else num_iterations
        )
        num_primal_dual_loops = (
            [num_primal_dual_loops]
            if not isinstance(num_primal_dual_loops, list)
            else num_primal_dual_loops
        )
        num_bregman_loops = (
            [num_bregman_loops]
            if not isinstance(num_bregman_loops, list)
            else num_bregman_loops
        )
        step_scale = (
            [step_scale] if not isinstance(step_scale, list) else step_scale
        )
        gamma = [gamma] if not isinstance(gamma, list) else gamma

        iterables_TV_parameters = [
            ("in_lambda", lambdaTV),
            ("in_deltat", deltatTV),
            ("in_iter", num_iterations),
            ("in_loop", num_primal_dual_loops),
            ("in_bregman_loop", num_bregman_loops),
            ("in_step_scale", step_scale),
            ("in_gamma", gamma),
        ]

        srtkTVSuperResolution.iterables = iterables_TV_parameters
    else:
        srtkTVSuperResolution.inputs.in_lambda = lambdaTV
        srtkTVSuperResolution.inputs.in_deltat = deltatTV

        srtkTVSuperResolution.inputs.in_iter = num_iterations
        srtkTVSuperResolution.inputs.in_loop = num_primal_dual_loops
        srtkTVSuperResolution.inputs.in_bregman_loop = num_bregman_loops
        srtkTVSuperResolution.inputs.in_step_scale = step_scale
        srtkTVSuperResolution.inputs.in_gamma = gamma

    if p_do_refine_hr_mask:
        srtkHRMask = pe.Node(
            interface=postprocess.MialsrtkRefineHRMaskByIntersection(
                verbose=p_verbose
            ),
            name="srtkHRMask",
        )
    else:
        srtkHRMask = pe.Node(
            interface=postprocess.BinarizeImage(), name="srtkHRMask"
        )

    if p_do_reconstruct_labels:
        recon_labels_stage = recon_labelmap_stage.create_recon_labelmap_stage(
            p_sub_ses=p_sub_ses, p_verbose=p_verbose
        )

    recon_stage.connect(
        inputnode,
        ("input_masks", utils.sort_ascending),
        srtkImageReconstruction,
        "input_masks",
    )
    recon_stage.connect(
        inputnode, "stacks_order", srtkImageReconstruction, "stacks_order"
    )

    if p_do_nlm_denoising:
        recon_stage.connect(
            inputnode,
            ("input_images_nlm", utils.sort_ascending),
            srtkImageReconstruction,
            "input_images",
        )

        recon_stage.connect(
            inputnode, "stacks_order", sdiComputation, "stacks_order"
        )
        recon_stage.connect(
            inputnode,
            ("input_images", utils.sort_ascending),
            sdiComputation,
            "input_images",
        )
        recon_stage.connect(
            inputnode,
            ("input_masks", utils.sort_ascending),
            sdiComputation,
            "input_masks",
        )
        recon_stage.connect(
            srtkImageReconstruction,
            ("output_transforms", utils.sort_ascending),
            sdiComputation,
            "input_transforms",
        )
        recon_stage.connect(
            srtkImageReconstruction,
            "output_sdi",
            sdiComputation,
            "input_reference",
        )

        recon_stage.connect(
            sdiComputation, "output_sdi", srtkTVSuperResolution, "input_sdi"
        )
    else:
        recon_stage.connect(
            inputnode,
            ("input_images", utils.sort_ascending),
            srtkImageReconstruction,
            "input_images",
        )
        recon_stage.connect(
            srtkImageReconstruction,
            "output_sdi",
            srtkTVSuperResolution,
            "input_sdi",
        )

    recon_stage.connect(
        inputnode,
        ("input_images", utils.sort_ascending),
        srtkTVSuperResolution,
        "input_images",
    )

    recon_stage.connect(
        srtkImageReconstruction,
        ("output_transforms", utils.sort_ascending),
        srtkTVSuperResolution,
        "input_transforms",
    )
    recon_stage.connect(
        inputnode,
        ("input_masks", utils.sort_ascending),
        srtkTVSuperResolution,
        "input_masks",
    )
    recon_stage.connect(
        inputnode, "stacks_order", srtkTVSuperResolution, "stacks_order"
    )

    if p_do_reconstruct_labels:
        recon_stage.connect(
            inputnode,
            ("input_labels", utils.sort_ascending),
            recon_labels_stage,
            "inputnode.input_labels",
        )
        recon_stage.connect(
            inputnode,
            ("input_masks", utils.sort_ascending),
            recon_labels_stage,
            "inputnode.input_masks",
        )
        recon_stage.connect(
            srtkImageReconstruction,
            ("output_transforms", utils.sort_ascending),
            recon_labels_stage,
            "inputnode.input_transforms",
        )

        recon_stage.connect(
            srtkImageReconstruction,
            "output_sdi",
            recon_labels_stage,
            "inputnode.input_reference",
        )
        recon_stage.connect(
            inputnode,
            "stacks_order",
            recon_labels_stage,
            "inputnode.stacks_order",
        )

        recon_stage.connect(
            recon_labels_stage,
            "outputnode.output_labelmap",
            outputnode,
            "output_labelmap",
        )

    if p_do_refine_hr_mask:
        recon_stage.connect(
            inputnode,
            ("input_images", utils.sort_ascending),
            srtkHRMask,
            "input_images",
        )
        recon_stage.connect(
            inputnode,
            ("input_masks", utils.sort_ascending),
            srtkHRMask,
            "input_masks",
        )
        recon_stage.connect(
            srtkImageReconstruction,
            ("output_transforms", utils.sort_ascending),
            srtkHRMask,
            "input_transforms",
        )
        recon_stage.connect(
            srtkImageReconstruction, "output_sdi", srtkHRMask, "input_sr"
        )
    else:
        recon_stage.connect(
            srtkImageReconstruction, "output_sdi", srtkHRMask, "input_image"
        )

    if p_do_nlm_denoising:
        recon_stage.connect(
            sdiComputation, "output_sdi", outputnode, "output_sdi"
        )
    else:
        recon_stage.connect(
            srtkImageReconstruction, "output_sdi", outputnode, "output_sdi"
        )

    recon_stage.connect(
        srtkImageReconstruction,
        "output_transforms",
        outputnode,
        "output_transforms",
    )
    recon_stage.connect(
        srtkTVSuperResolution, "output_sr", outputnode, "output_sr"
    )
    recon_stage.connect(
        srtkHRMask, "output_srmask", outputnode, "output_hr_mask"
    )
    recon_stage.connect(
        srtkTVSuperResolution,
        "output_json_path",
        outputnode,
        "output_json_path",
    )
    recon_stage.connect(
        srtkTVSuperResolution, "output_sr_png", outputnode, "output_sr_png"
    )
    recon_stage.connect(
        srtkTVSuperResolution,
        "output_TV_parameters",
        outputnode,
        "output_TV_parameters",
    )

    if p_do_multi_parameters:
        outputnode.inputs.output_TV_params = iterables_TV_parameters

    return recon_stage, srtkTVSuperResolution.name
