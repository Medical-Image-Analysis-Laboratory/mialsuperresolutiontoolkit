# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
# This software is distributed under the open-source license Modified BSD.

"""Module for the reconstruction stage of the super-resolution
reconstruction pipeline."""

from traits.api import *
from nipype.interfaces.base import (TraitedSpec, File,
                                    InputMultiPath, OutputMultiPath,
                                    BaseInterface, BaseInterfaceInputSpec)
from nipype.interfaces import utility as util
from nipype.interfaces.io import DataGrabber
from nipype.pipeline import engine as pe

import pymialsrtk.workflows.recon_labelmap_stage as recon_labelmap_stage

import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.utils as utils


def create_recon_stage(p_paramTV,
                       p_use_manual_masks,
                       p_do_nlm_denoising=False,
                       p_do_reconstruct_labels=False,
                       p_do_refine_hr_mask=False,
                       p_skip_svr=False,
                       p_sub_ses='',
                       p_verbose=False,
                       name="recon_stage"):
    """Create a super-resolution reconstruction workflow
    Parameters
    ----------
    ::
        name : name of workflow (default: recon_stage)
        p_use_manual_masks :
        p_do_nlm_denoising : weither to proceed to non-local mean denoising
        p_do_reconstruct_labels :
        p_do_refine_hr_mask :
        p_skip_svr :
        p_sub_ses :
    Inputs::
        inputnode.input_images : Input T2w images (list of filenames)
        inputnode.input_images_nlm : Input T2w images (list of filenames),
            if p_do_nlm_denoising was set (list of filenames)
        inputnode.input_masks : Input mask images (list of filenames)
        inputnode.stacks_order : Order of stacks in the reconstruction
            (list of integer)
    Outputs::
        outputnode.output_sr : SR reconstructed image (filename)
        outputnode.output_sdi : SDI image (filename)
        outputnode.output_hr_mask : SRR mask (filename)
        outputnode.output_tranforms : Transfmation estimated parameters
            (list of filenames)
    Example
    -------
    >>> recon_stage = create_preproc_stage(p_do_nlm_denoising=False)
    >>> recon_stage.inputs.inputnode.input_images =
            ['sub-01_run-1_T2w.nii.gz', 'sub-01_run-2_T2w.nii.gz']
    >>> recon_stage.inputs.inputnode.input_masks =
            ['sub-01_run-1_T2w_mask.nii.gz', 'sub-01_run-2_T2w_mask.nii.gz']
    >>> recon_stage.inputs.inputnode.p_do_nlm_denoising = 'mask.nii'
    >>> recon_stage.run() # doctest: +SKIP
    """

    recon_stage = pe.Workflow(name=name)

    # Set up a node to define all inputs required for the
    # preprocessing workflow
    input_fields = ['input_images', 'input_masks', 'stacks_order']

    if p_do_nlm_denoising:
        input_fields += ['input_images_nlm']

    if p_do_reconstruct_labels:
        input_fields += ['input_labels']

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    output_fields = ['output_sr', 'output_sdi',
                     'output_hr_mask', 'output_transforms',
                     'output_json_path',
                     'output_sr_png']

    if p_do_reconstruct_labels:
        output_fields += ['output_labelmap']

    outputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=output_fields),
        name='outputnode')

    deltatTV = p_paramTV["deltatTV"] \
        if "deltatTV" in p_paramTV.keys() else 0.01
    lambdaTV = p_paramTV["lambdaTV"] \
        if "lambdaTV" in p_paramTV.keys() else 0.75

    num_iterations = p_paramTV["num_iterations"] \
        if "num_iterations" in p_paramTV.keys() else 50
    num_primal_dual_loops = p_paramTV["num_primal_dual_loops"] \
        if "num_primal_dual_loops" in p_paramTV.keys() else 10
    num_bregman_loops = p_paramTV["num_bregman_loops"] \
        if "num_bregman_loops" in p_paramTV.keys() else 3
    step_scale = p_paramTV["step_scale"] \
        if "step_scale" in p_paramTV.keys() else 1
    gamma = p_paramTV["gamma"] if "gamma" in p_paramTV.keys() else 1

    srtkImageReconstruction = pe.Node(
        interface=reconstruction.MialsrtkImageReconstruction(),
        name='srtkImageReconstruction')
    srtkImageReconstruction.inputs.sub_ses = p_sub_ses
    srtkImageReconstruction.inputs.no_reg = p_skip_svr
    srtkImageReconstruction.inputs.verbose = p_verbose

    if p_do_nlm_denoising:
        sdiComputation = pe.Node(
            interface=reconstruction.MialsrtkSDIComputation(),
            name='sdiComputation')
        sdiComputation.inputs.sub_ses = p_sub_ses
        sdiComputation.inputs.verbose = p_verbose

    srtkTVSuperResolution = pe.Node(
        interface=reconstruction.MialsrtkTVSuperResolution(),
        name='srtkTVSuperResolution')
    srtkTVSuperResolution.inputs.sub_ses = p_sub_ses
    srtkTVSuperResolution.inputs.use_manual_masks = p_use_manual_masks
    srtkTVSuperResolution.inputs.in_iter = num_iterations
    srtkTVSuperResolution.inputs.in_loop = num_primal_dual_loops
    srtkTVSuperResolution.inputs.in_bregman_loop = num_bregman_loops
    srtkTVSuperResolution.inputs.in_step_scale = step_scale
    srtkTVSuperResolution.inputs.in_gamma = gamma
    srtkTVSuperResolution.inputs.in_deltat = deltatTV
    srtkTVSuperResolution.inputs.in_lambda = lambdaTV
    srtkTVSuperResolution.inputs.verbose = p_verbose


    if p_do_refine_hr_mask:
        srtkHRMask = pe.Node(
            interface=postprocess.MialsrtkRefineHRMaskByIntersection(),
            name='srtkHRMask')
        srtkHRMask.inputs.verbose = p_verbose
    else:
        srtkHRMask = pe.Node(interface=postprocess.BinarizeImage(),
                             name='srtkHRMask')

    if p_do_reconstruct_labels:
        recon_labels_stage = recon_labelmap_stage.create_recon_labelmap_stage(
            sub_ses=p_sub_ses)
        # recon_labels_stage.inputs.inputnode.label_ids = \
        #     [0, 1, 2, 3, 4, 5, 6, 7]


    recon_stage.connect(inputnode, "input_masks",
                        srtkImageReconstruction, "input_masks")
    recon_stage.connect(inputnode, "stacks_order",
                        srtkImageReconstruction, "stacks_order")

    if p_do_nlm_denoising:
        recon_stage.connect(inputnode, "input_images_nlm",
                            srtkImageReconstruction, "input_images")

        recon_stage.connect(inputnode, "stacks_order",
                            sdiComputation, "stacks_order")
        recon_stage.connect(inputnode, "input_images",
                            sdiComputation, "input_images")
        recon_stage.connect(inputnode, "input_masks",
                            sdiComputation, "input_masks")
        recon_stage.connect(srtkImageReconstruction, "output_transforms",
                            sdiComputation, "input_transforms")
        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            sdiComputation, "input_reference")

        recon_stage.connect(sdiComputation, "output_hr",
                            srtkTVSuperResolution, "input_sdi")
    else:
        recon_stage.connect(inputnode, "input_images",
                            srtkImageReconstruction, "input_images")
        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            srtkTVSuperResolution, "input_sdi")

    recon_stage.connect(inputnode, "input_images",
                        srtkTVSuperResolution, "input_images")

    recon_stage.connect(srtkImageReconstruction, "output_transforms",
                        srtkTVSuperResolution, "input_transforms")
    recon_stage.connect(inputnode, "input_masks",
                        srtkTVSuperResolution, "input_masks")
    recon_stage.connect(inputnode, "stacks_order",
                        srtkTVSuperResolution, "stacks_order")

    if p_do_reconstruct_labels:
        recon_stage.connect(inputnode, "input_labels",
                            recon_labels_stage, "inputnode.input_labels")
        recon_stage.connect(inputnode, "input_masks",
                            recon_labels_stage, "inputnode.input_masks")
        recon_stage.connect(srtkImageReconstruction, "output_transforms",
                            recon_labels_stage, "inputnode.input_transforms")

        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            recon_labels_stage, "inputnode.input_reference")
        recon_stage.connect(inputnode, "stacks_order",
                            recon_labels_stage, "inputnode.stacks_order")

        recon_stage.connect(recon_labels_stage,
                            "outputnode.output_labelmap",
                            outputnode, "output_labelmap")

    if p_do_refine_hr_mask:
        recon_stage.connect(inputnode, "input_images",
                            srtkHRMask, "input_images")

        recon_stage.connect(inputnode, "input_masks",
                            srtkHRMask, "input_masks")
        recon_stage.connect(srtkImageReconstruction, ("output_transforms",
                                                      utils.sort_ascending),
                            srtkHRMask, "input_transforms")
        recon_stage.connect(srtkTVSuperResolution, "output_sr",
                            srtkHRMask, "input_sr")
    else:
        recon_stage.connect(srtkTVSuperResolution, "output_sr",
                            srtkHRMask, "input_image")

    if p_do_nlm_denoising:
        recon_stage.connect(sdiComputation, "output_hr",
                            outputnode, "output_sdi")
    else:
        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            outputnode, "output_sdi")

    recon_stage.connect(srtkImageReconstruction, "output_transforms",
                        outputnode, "output_transforms")
    recon_stage.connect(srtkTVSuperResolution, "output_sr",
                        outputnode, "output_sr")
    recon_stage.connect(srtkHRMask, "output_srmask",
                        outputnode, "output_hr_mask")
    recon_stage.connect(srtkTVSuperResolution, "output_json_path",
                        outputnode, "output_json_path")
    recon_stage.connect(srtkTVSuperResolution, "output_sr_png",
                        outputnode, "output_sr_png")

    return recon_stage
