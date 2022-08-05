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
from nipype.pipeline import engine as pe

from nipype.interfaces.io import DataGrabber

import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.utils as utils

def create_recon_stage(p_paramTV,
                       p_use_manual_masks,
                       p_do_nlm_denoising=False,
                       p_do_refine_hr_mask=False,
                       p_skip_svr=False,
                       p_do_anat_orientation=False,
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

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=
                                         ['output_sr', 'output_sdi',
                                          'output_hr_mask',
                                          'output_transforms',
                                          'output_json_path',
                                          'output_sr_png',
                                          'output_sr_aligned']),
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

    if p_do_nlm_denoising:
        # srtkMaskImage01_nlm = pe.MapNode(
        #     interface=preprocess.MialsrtkMaskImage(),
        #     name='srtkMaskImage01_nlm',
        #     iterfield=['in_file', 'in_mask'])

        sdiComputation = pe.Node(
            interface=reconstruction.MialsrtkSDIComputation(),
            name='sdiComputation')
        sdiComputation.inputs.sub_ses = p_sub_ses

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


    if p_do_refine_hr_mask:
        srtkHRMask = pe.Node(
            interface=postprocess.MialsrtkRefineHRMaskByIntersection(),
            name='srtkHRMask')
    else:
        srtkHRMask = pe.Node(interface=postprocess.BinarizeImage(),
                             name='srtkHRMask')

    if p_do_anat_orientation and not p_do_nlm_denoising:
        if 'chuv012' in p_sub_ses:
            ga = 22 # chuv012
        else:
            ga = 30 # chuv026
        ga_str = str(ga) + 'exp' if ga > 35 else str(ga)

        atlas_grabber = pe.Node(
            interface=DataGrabber(outfields=['atlas', 'tissue']),
            name='atlas_grabber'
        )
        atlas_grabber.inputs.base_directory = '/sta'
        atlas_grabber.inputs.template = '*'
        atlas_grabber.inputs.raise_on_empty = False
        atlas_grabber.inputs.sort_filelist = True

        atlas_grabber.inputs.field_template = dict(atlas='STA'+ga_str+'.nii.gz')

        resample_t2w_template = pe.Node(interface=preprocess.ResampleImage(),
                             name='resample_t2w_template')

        align_volume = pe.Node(interface=preprocess.AlignImageToReference(),
                             name='align_volume')
        align_volume.config = {'execution': {'keep_unnecessary_outputs': 'true'}}

        compose_transforms = pe.MapNode(interface=preprocess.ComposeTransforms(),
                             name='compose_transforms',
                             iterfield=['input_svr_from_mial', 'input_LR'])
        compose_transforms.config = {'execution': {'keep_unnecessary_outputs': 'true'}}

        sdiComputation_anat = pe.Node(
            interface=reconstruction.MialsrtkSDIComputation(),
            name='sdiComputation_anat')
        sdiComputation_anat.inputs.sub_ses = p_sub_ses

        srtkTVSuperResolution_anat = pe.Node(
            interface=reconstruction.MialsrtkTVSuperResolution(),
            name='srtkTVSuperResolution_anat')
        srtkTVSuperResolution_anat.inputs.sub_ses = p_sub_ses
        srtkTVSuperResolution_anat.inputs.use_manual_masks = p_use_manual_masks

        srtkTVSuperResolution_anat.inputs.in_iter = num_iterations
        srtkTVSuperResolution_anat.inputs.in_loop = num_primal_dual_loops
        srtkTVSuperResolution_anat.inputs.in_bregman_loop = num_bregman_loops
        srtkTVSuperResolution_anat.inputs.in_step_scale = step_scale
        srtkTVSuperResolution_anat.inputs.in_gamma = gamma
        srtkTVSuperResolution_anat.inputs.in_deltat = deltatTV
        srtkTVSuperResolution_anat.inputs.in_lambda = lambdaTV


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
    recon_stage.connect(inputnode, "stacks_order",
                        srtkTVSuperResolution, "stacks_order")

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
        recon_stage.connect(sdiComputation, "output_sdi",
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



    if p_do_anat_orientation and not p_do_nlm_denoising:
        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            resample_t2w_template, "input_reference")
        recon_stage.connect(atlas_grabber, "atlas",
                            resample_t2w_template, "input_image")


        recon_stage.connect(srtkImageReconstruction, "output_sdi",
                            align_volume, "input_image")
        recon_stage.connect(resample_t2w_template, "output_image",
                            align_volume, "input_template")

        recon_stage.connect(align_volume, "output_transform",
                            compose_transforms, "input_rigid")

        recon_stage.connect(inputnode, "input_images",
                            compose_transforms, "input_LR")

        recon_stage.connect(srtkImageReconstruction, ("output_transforms",
                                                      utils.sort_ascending),
                            compose_transforms, "input_svr_from_mial")



        recon_stage.connect(inputnode, "stacks_order",
                            sdiComputation_anat, "stacks_order")
        recon_stage.connect(inputnode, "input_images",
                            sdiComputation_anat, "input_images")
        recon_stage.connect(inputnode, "input_masks",
                            sdiComputation_anat, "input_masks")
        recon_stage.connect(compose_transforms, "output_transform",
                            sdiComputation_anat, "input_transforms")
        recon_stage.connect(align_volume, "output_image",
                            sdiComputation_anat, "input_reference")


        recon_stage.connect(sdiComputation_anat, "output_sdi",
                            srtkTVSuperResolution_anat, "input_sdi")

        recon_stage.connect(inputnode, "input_images",
                            srtkTVSuperResolution_anat, "input_images")

        recon_stage.connect(compose_transforms, "output_transform",
                            srtkTVSuperResolution_anat, "input_transforms")
        recon_stage.connect(inputnode, "input_masks",
                            srtkTVSuperResolution_anat, "input_masks")
        recon_stage.connect(inputnode, "stacks_order",
                            srtkTVSuperResolution_anat, "stacks_order")

        recon_stage.connect(srtkTVSuperResolution_anat, "output_sr",
                            outputnode, "output_sr_aligned") # Temporary
        ##

    return recon_stage
