# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the reconstruction stage of the super-resolution reconstruction pipeline."""

import os
import traceback
from glob import glob
import pathlib

from traits.api import *

from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.utils as utils

from nipype import config
from nipype import logging as nipype_logging

from nipype.pipeline import engine as pe
from nipype.interfaces import utility as util


def create_recon_stage(p_paramTV,
                       p_use_manual_masks,
                       p_do_nlm_denoising = False,
                       p_do_refine_hr_mask=False,
                       p_skip_svr=False,
                       p_bids_dir='',
                       p_sub_ses='',
                       name="recon_stage"):
    """Create a super-resolution reconstruction workflow
    Parameters
    ----------
    ::
        name : name of workflow (default: recon_stage)
        p_do_nlm_denoising : weither to proceed to non-local mean denoising
    Inputs::
        inputnode.input_images : Input T2w images (list of filenames)
        inputnode.input_images_nlm : Input T2w images (list of filenames), if p_do_nlm_denoising was set (list of filenames)
        inputnode.input_masks : Input mask images (list of filenames)
        inputnode.stacks_order : Order of stacks in the reconstruction (list of integer)
    Outputs::
        outputnode.output_sr : SR reconstructed image (filename)
        outputnode.output_sdi : SDI image (filename)
        outputnode.output_hr_mask : SRR mask (filename)
        outputnode.output_tranforms : Transfmation estimated parameters (list of filenames)
    Example
    -------
    >>> recon_stage = create_preproc_stage(bids_dir='/path/to/bids_dir', p_do_nlm_denoising=False)
    >>> recon_stage.inputs.inputnode.input_images = ['sub-01_run-1_T2w.nii.gz', 'sub-01_run-2_T2w.nii.gz']
    >>> recon_stage.inputs.inputnode.input_masks = ['sub-01_run-1_T2w_mask.nii.gz', 'sub-01_run-2_T2w_mask.nii.gz']
    >>> recon_stage.inputs.inputnode.p_do_nlm_denoising = 'mask.nii'
    >>> recon_stage.run() # doctest: +SKIP
    """

    recon_stage = pe.Workflow(name=name)
    """
    Set up a node to define all inputs required for the preprocessing workflow
    """
    input_fields = ['input_images', 'input_masks', 'stacks_order']

    if p_do_nlm_denoising:
        input_fields += ['input_images_nlm']

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=
                                         ['output_sr', 'output_sdi',
                                          'output_hr_mask', 'output_transforms',
                                          'output_json_path',
                                          'output_sr_png']),
        name='outputnode')

    """
    """

    deltatTV = p_paramTV["deltatTV"] if "deltatTV" in p_paramTV.keys() else 0.01
    lambdaTV = p_paramTV["lambdaTV"] if "lambdaTV" in p_paramTV.keys() else 0.75

    num_iterations = p_paramTV["num_iterations"] if "num_iterations" in p_paramTV.keys() else 50
    num_primal_dual_loops = p_paramTV["num_primal_dual_loops"] if "num_primal_dual_loops" in p_paramTV.keys() else 10
    num_bregman_loops = p_paramTV["num_bregman_loops"] if "num_bregman_loops" in p_paramTV.keys() else 3
    step_scale = p_paramTV["step_scale"] if "step_scale" in p_paramTV.keys() else 1
    gamma = p_paramTV["gamma"] if "gamma" in p_paramTV.keys() else 1

    srtkImageReconstruction = pe.Node(interface=reconstruction.MialsrtkImageReconstruction(),
                                   name='srtkImageReconstruction')
    srtkImageReconstruction.inputs.bids_dir = p_bids_dir
    srtkImageReconstruction.inputs.sub_ses = p_sub_ses
    srtkImageReconstruction.inputs.no_reg = p_skip_svr

    if p_do_nlm_denoising:
        # srtkMaskImage01_nlm = pe.MapNode(
        #     interface=preprocess.MialsrtkMaskImage(),
        #     name='srtkMaskImage01_nlm',
        #     iterfield=['in_file', 'in_mask'])
        # srtkMaskImage01_nlm.inputs.bids_dir = p_bids_dir

        sdiComputation = pe.Node(
            interface=reconstruction.MialsrtkSDIComputation(),
            name='sdiComputation')
        sdiComputation.inputs.sub_ses = p_sub_ses

    srtkTVSuperResolution = pe.Node(interface=reconstruction.MialsrtkTVSuperResolution(),
                                 name='srtkTVSuperResolution')
    srtkTVSuperResolution.inputs.bids_dir = p_bids_dir
    srtkTVSuperResolution.inputs.sub_ses = p_sub_ses
    srtkTVSuperResolution.inputs.use_manual_masks = p_use_manual_masks

    srtkTVSuperResolution.inputs.in_iter = num_iterations
    srtkTVSuperResolution.inputs.in_loop = num_primal_dual_loops
    srtkTVSuperResolution.inputs.in_bregman_loop = num_bregman_loops
    srtkTVSuperResolution.inputs.in_step_scale = step_scale
    srtkTVSuperResolution.inputs.in_gamma = gamma
    srtkTVSuperResolution.inputs.in_deltat = deltatTV
    srtkTVSuperResolution.inputs.in_lambda = lambdaTV

    if p_do_refine_hr_mask:
        srtkHRMask = pe.Node(interface=postprocess.MialsrtkRefineHRMaskByIntersection(),
                             name='srtkHRMask')
        srtkHRMask.inputs.bids_dir = p_bids_dir
    else:
        srtkHRMask = pe.Node(interface=postprocess.BinarizeImage(),
                             name='srtkHRMask')

    recon_stage.connect(inputnode, "input_masks",
                        srtkImageReconstruction, "input_masks")
    recon_stage.connect(inputnode, "stacks_order",
                        srtkImageReconstruction, "stacks_order")

    if p_do_nlm_denoising:
        recon_stage.connect(inputnode, "input_images_nlm",
                        srtkImageReconstruction, "input_images")

        recon_stage.connect(inputnode, "stacks_order",
                            sdiComputation, "stacks_order")
        recon_stage.connect(inputnode, "input_images_nlm",
                            sdiComputation, "input_images")
        recon_stage.connect(inputnode, "input_masks",
                            sdiComputation, "input_masks")
        recon_stage.connect(srtkImageReconstruction, "output_transforms",
                            sdiComputation, "input_transforms")
        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            sdiComputation, "input_reference")

        recon_stage.connect(sdiComputation, "output_sdi",
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
    recon_stage.connect(inputnode, "stacks_order", srtkTVSuperResolution, "stacks_order")

    if p_do_refine_hr_mask:
        recon_stage.connect(inputnode, "input_images",srtkHRMask, "input_images")

        recon_stage.connect(inputnode, "input_masks", srtkHRMask, "input_masks")
        recon_stage.connect(srtkImageReconstruction, ("output_transforms",
                                                  utils.sort_ascending),
                        srtkHRMask, "input_transforms")
        recon_stage.connect(srtkTVSuperResolution, "output_sr",
                            srtkHRMask, "input_sr")
    else:
        recon_stage.connect(srtkTVSuperResolution, "output_sr",
                            srtkHRMask, "input_image")

    if p_do_nlm_denoising:
        recon_stage.connect(sdiComputation, "output_sdi",
                            outputnode, "output_sdi")
    else:
        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            outputnode, "output_sdi")
    recon_stage.connect(srtkTVSuperResolution, "output_sr",
                        outputnode, "output_sr")
    recon_stage.connect(srtkHRMask, "output_srmask",
                        outputnode, "output_hr_mask")
    recon_stage.connect(srtkTVSuperResolution, "output_json_path",
                        outputnode, "output_json_path")
    recon_stage.connect(srtkTVSuperResolution, "output_sr_png",
                        outputnode, "output_sr_png")

    return recon_stage
