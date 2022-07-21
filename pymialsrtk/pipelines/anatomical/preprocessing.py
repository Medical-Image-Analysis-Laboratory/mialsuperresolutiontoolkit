# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.
# R. What I'd like to do -> I'd like to have everything up to the SR reconstruction and see
# if skipping it makes the overall pipeline faster, and if I can then use these to explore
# compute some downstream metrics.
"""Module for the preprocessing of the super-reconstruction pipeline."""

import os
import sys
import platform
import json
import shutil
import pkg_resources
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from jinja2 import __version__ as __jinja2_version__

import nibabel as nib

from nipype.info import __version__ as __nipype_version__
from nipype import config
from nipype import logging as nipype_logging
from nipype.interfaces.io import DataGrabber, DataSink

from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.utils as utils
from nipype.interfaces.utility import IdentityInterface

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.workflows.preproc_stage as preproc_stage
import pymialsrtk.workflows.srr_output_stage as srr_output_stage
from pymialsrtk.bids.utils import write_bids_derivative_description

# Get pymialsrtk version
from pymialsrtk.info import __version__

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


def create_registration_stage(p_do_nlm_denoising = False,
                       p_skip_svr=False,
                       p_sub_ses='',
                       name="registration_stage"):
    """Create a a registration workflow
    Parameters
    ----------
    ::
        name : name of workflow (default: registration_stage)
        p_do_nlm_denoising : weither to proceed to non-local mean denoising
    Inputs::
        inputnode.input_images : Input T2w images (list of filenames)
        inputnode.input_images_nlm : Input T2w images (list of filenames), if p_do_nlm_denoising was set (list of filenames)
        inputnode.input_masks : Input mask images (list of filenames)
        inputnode.stacks_order : Order of stacks in the registration (list of integer)
    Outputs::
        outputnode.output_sdi : SDI image (filename)
        outputnode.output_tranforms : Transfmation estimated parameters (list of filenames)
    Example
    -------
    >>> registration_stage = create_preproc_stage(p_do_nlm_denoising=False)
    >>> registration_stage.inputs.inputnode.input_images = ['sub-01_run-1_T2w.nii.gz', 'sub-01_run-2_T2w.nii.gz']
    >>> registration_stage.inputs.inputnode.input_masks = ['sub-01_run-1_T2w_mask.nii.gz', 'sub-01_run-2_T2w_mask.nii.gz']
    >>> registration_stage.inputs.inputnode.p_do_nlm_denoising = 'mask.nii'
    >>> registration_stage.run() # doctest: +SKIP
    """

    registration_stage = pe.Workflow(name=name)
    """
    Set up a node to define all inputs required for the preprocessing workflow
    """
    input_fields = ['input_images', 'input_masks', 'stacks_order']

    if p_do_nlm_denoising:
        input_fields += ['input_images_nlm']

    # Input node with the input fields specified above
    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=input_fields),
        name='inputnode')

    # Output node with the interpolated HR image + transforms from registration
    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=
                                         ['output_sdi', 'output_transforms']),
        name='outputnode')

    """
    """

    srtkImageReconstruction = pe.Node(
        interface=reconstruction.MialsrtkImageReconstruction(),
        name='srtkImageReconstruction')
    srtkImageReconstruction.inputs.sub_ses = p_sub_ses
    srtkImageReconstruction.inputs.no_reg = p_skip_svr

    if p_do_nlm_denoising:
        sdiComputation = pe.Node(
            interface=reconstruction.MialsrtkSDIComputation(),
            name='sdiComputation')
        sdiComputation.inputs.sub_ses = p_sub_ses


    registration_stage.connect(inputnode, "input_masks",
                        srtkImageReconstruction, "input_masks")
    registration_stage.connect(inputnode, "stacks_order",
                        srtkImageReconstruction, "stacks_order")

    if p_do_nlm_denoising:
        registration_stage.connect(inputnode, "input_images_nlm",
                            srtkImageReconstruction, "input_images")

        registration_stage.connect(inputnode, "stacks_order",
                            sdiComputation, "stacks_order")
        registration_stage.connect(inputnode, "input_images_nlm",
                            sdiComputation, "input_images")
        registration_stage.connect(inputnode, "input_masks",
                            sdiComputation, "input_masks")
        registration_stage.connect(srtkImageReconstruction, "output_transforms",
                            sdiComputation, "input_transforms")
        registration_stage.connect(srtkImageReconstruction, "output_sdi",
                            sdiComputation, "input_reference")

    else:
        registration_stage.connect(inputnode, "input_images",
                            srtkImageReconstruction, "input_images")

    if p_do_nlm_denoising:
        registration_stage.connect(sdiComputation, "output_sdi",
                            outputnode, "output_sdi")
    else:
        registration_stage.connect(srtkImageReconstruction, "output_sdi",
                            outputnode, "output_sdi")

    registration_stage.connect(srtkImageReconstruction, "output_transforms",
                        outputnode, "output_transforms")

    return registration_stage


def create_prepro_output_stage(p_do_nlm_denoising=False,
                            p_skip_stacks_ordering=False,
                            name="prepro_output_stage"):
    """Create an output management workflow
    for the preprocessing only.
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

    prepro_output_stage = pe.Workflow(name=name)
    """
    Set up a node to define all inputs required for the srr output workflow
    """
    input_fields = ["sub_ses", "sr_id", "stacks_order", "use_manual_masks", "final_res_dir"]
    input_fields += ["input_masks", "input_images", "input_sdi", "input_transforms"]

    if not p_skip_stacks_ordering:
        input_fields += ['report_image', 'motion_tsv']
    if p_do_nlm_denoising:
        input_fields += ['input_images_nlm']

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

    prepro_output_stage.connect(inputnode, "sub_ses", finalFilenamesGeneration, "sub_ses")
    prepro_output_stage.connect(inputnode, "sr_id", finalFilenamesGeneration, "sr_id")
    prepro_output_stage.connect(inputnode, "stacks_order", finalFilenamesGeneration, "stacks_order")
    prepro_output_stage.connect(inputnode, "use_manual_masks", finalFilenamesGeneration, "use_manual_masks")

    prepro_output_stage.connect(finalFilenamesGeneration, "substitutions",
                    datasink, "substitutions")

    prepro_output_stage.connect(inputnode, "final_res_dir", datasink, 'base_directory')

    if not p_skip_stacks_ordering:
        prepro_output_stage.connect(inputnode, "report_image",
                                 datasink, 'figures.@stackOrderingQC')
        prepro_output_stage.connect(inputnode, "motion_tsv",
                                 datasink, 'anat.@motionTSV')
    prepro_output_stage.connect(inputnode, "input_masks",
                             datasink, 'anat.@LRmasks')
    prepro_output_stage.connect(inputnode, "input_images",
                             datasink, 'anat.@LRsPreproc')
    prepro_output_stage.connect(inputnode, "input_transforms",
                             datasink, 'xfm.@transforms')

    prepro_output_stage.connect(inputnode, "input_sdi",
                             datasink, 'anat.@SDI')
    if p_do_nlm_denoising:
        prepro_output_stage.connect(inputnode, "input_images_nlm",
                                 datasink, 'anat.@LRsDenoised')

    return prepro_output_stage


class PreprocessingPipeline:
    """Class used to represent the workflow of the preprocessing for
    the Super-Resolution reconstruction pipeline.

    Attributes
    -----------
    bids_dir : string
        BIDS root directory (required)

    output_dir : string
        Output derivatives directory (required)

    subject : string
        Subject ID (in the form ``sub-XX``)

    wf : nipype.pipeline.Workflow
        Nipype workflow of the reconstruction pipeline

    deltatTV : string
        Super-resolution optimization time-step

    lambdaTV : float
        Regularization weight (default is 0.75)

    num_iterations : string
        Number of iterations in the primal/dual loops used in the optimization of the total-variation
        super-resolution algorithm.

    num_primal_dual_loops : string
        Number of primal/dual (inner) loops used in the optimization of the total-variation
        super-resolution algorithm.

    num_bregman_loops : string
        Number of Bregman (outer) loops used in the optimization of the total-variation
        super-resolution algorithm.

    step_scale : string
        Step scale parameter used in the optimization of the total-variation
        super-resolution algorithm.

    gamma : string
        Gamma parameter used in the optimization of the total-variation
        super-resolution algorithm.

    sr_id : string
        ID of the reconstruction useful to distinguish when multiple reconstructions
        with different order of stacks are run on the same subject

    session : string
        Session ID if applicable (in the form ``ses-YY``)

    m_stacks : list(int)
        List of stack to be used in the reconstruction.
        The specified order is kept if `skip_stacks_ordering` is True.

    m_masks_derivatives_dir : string
        directory basename in BIDS directory derivatives where to search for masks (optional)

    m_skip_svr : bool
        Weither the Slice-to-Volume Registration should be skipped in the image reconstruction.
        (default is False)

    m_do_refine_hr_mask : bool
        Weither a refinement of the HR mask should be performed. (default is False)

    m_do_nlm_denoising : bool
        Weither the NLM denoising preprocessing should be performed prior to motion estimation. (default is False)

    m_skip_stacks_ordering : bool (optional)
        Weither the automatic stacks ordering should be skipped. (default is False)

    Examples
    --------
    >>> from pymialsrtk.pipelines.anatomical.srr import AnatomicalPipeline
    >>> # Create a new instance
    >>> pipeline = AnatomicalPipeline(bids_dir='/path/to/bids_dir',
                                      output_dir='/path/to/output_dir',
                                      subject='sub-01',
                                      p_stacks=[1,3,2,0],
                                      sr_id=1,
                                      session=None,
                                      paramTV={deltatTV = "0.001",
                                               lambdaTV = "0.75",
                                               num_primal_dual_loops = "20"},
                                      masks_derivatives_dir="/custom/mask_dir",
                                      masks_desc=None,
                                      p_dict_custom_interfaces=None)
    >>> # Create the super resolution Nipype workflow
    >>> pipeline.create_workflow()
    >>> # Execute the workflow
    >>> res = pipeline.run(number_of_cores=1) # doctest: +SKIP

    """

    pipeline_name = "preprocessing_pipeline"
    run_start_time = None
    run_end_time = None
    run_elapsed_time = None

    bids_dir = None
    output_dir = None
    subject = None
    wf = None
    sr_id = None
    session = None

    deltatTV = None
    lambdaTV = None
    num_iterations = None
    num_primal_dual_loops = None
    num_bregman_loops = None
    step_scale = None
    gamma = None

    m_stacks = None

    # Custom interfaces options
    m_skip_svr = None
    m_do_nlm_denoising = None
    m_skip_stacks_ordering = None
    m_do_refine_hr_mask = None

    m_masks_derivatives_dir = None
    use_manual_masks = False
    m_masks_desc = None

    openmp_number_of_cores = None
    nipype_number_of_cores = None

    def __init__(
        self, bids_dir, output_dir, subject, p_stacks=None, sr_id=1,
        session=None, paramTV=None, p_masks_derivatives_dir=None, p_masks_desc=None,
        p_dict_custom_interfaces=None,
        openmp_number_of_cores=None, nipype_number_of_cores=None
    ):
        """Constructor of AnatomicalPipeline class instance."""

        # BIDS processing parameters
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.subject = subject
        self.sr_id = sr_id
        self.session = session
        self.m_stacks = p_stacks

        self.openmp_number_of_cores = openmp_number_of_cores
        self.nipype_number_of_cores = nipype_number_of_cores

        # (default) sr tv parameters
        if paramTV is None:
            paramTV = dict()
        self.paramTV = paramTV

        # Use manual/custom brain masks
        # If masks directory is not specified use the automated brain extraction method.
        self.m_masks_derivatives_dir = p_masks_derivatives_dir
        self.use_manual_masks = True if self.m_masks_derivatives_dir is not None else False
        self.m_masks_desc = p_masks_desc if self.use_manual_masks else None

        # Custom interfaces and default values.
        if p_dict_custom_interfaces is not None:
            self.m_skip_svr = p_dict_custom_interfaces['skip_svr'] if 'skip_svr' in p_dict_custom_interfaces.keys() else False
            self.m_do_refine_hr_mask = p_dict_custom_interfaces['do_refine_hr_mask'] if 'do_refine_hr_mask' in p_dict_custom_interfaces.keys() else False
            self.m_do_nlm_denoising = p_dict_custom_interfaces['do_nlm_denoising'] if 'do_nlm_denoising' in p_dict_custom_interfaces.keys() else False

            self.m_skip_stacks_ordering = p_dict_custom_interfaces['skip_stacks_ordering'] if \
                ((self.m_stacks is not None) and ('skip_stacks_ordering' in p_dict_custom_interfaces.keys())) else False
        else:
            self.m_skip_svr = False
            self.m_do_refine_hr_mask = False
            self.m_do_nlm_denoising =  False
            self.m_skip_stacks_ordering = False

    def create_workflow(self):
        """Create the Niype workflow of the preprocessing pipeline.

        It is composed of a succession of Nodes and their corresponding parameters,
        where the output of node i goes to the input of node i+1.

        """
        sub_ses = self.subject
        if self.session is not None:
            sub_ses = ''.join([sub_ses, '_', self.session])

        if self.session is None:
            wf_base_dir = os.path.join(self.output_dir,
                                       '-'.join(["nipype", __nipype_version__]),
                                       self.subject,
                                       "rec-{}".format(self.sr_id))
            final_res_dir = os.path.join(self.output_dir,
                                         '-'.join(["pymialsrtk", __version__]),
                                         self.subject)
        else:
            wf_base_dir = os.path.join(self.output_dir,
                                       '-'.join(["nipype", __nipype_version__]),
                                       self.subject,
                                       self.session,
                                       "rec-{}".format(self.sr_id))
            final_res_dir = os.path.join(self.output_dir,
                                         '-'.join(["pymialsrtk", __version__]),
                                         self.subject,
                                         self.session)

        if not os.path.exists(wf_base_dir):
            os.makedirs(wf_base_dir)
        print("Process directory: {}".format(wf_base_dir))

        # Initialization (Not sure we can control the name of nipype log)
        if os.path.isfile(os.path.join(wf_base_dir, "prepocessing_pypeline.log")):
            os.unlink(os.path.join(wf_base_dir, "prepocessing_pypeline.log"))

        self.wf = pe.Workflow(name=self.pipeline_name,base_dir=wf_base_dir)

        config.update_config(
            {
                'logging': {
                      'log_directory': os.path.join(wf_base_dir),
                      'log_to_file': True
                },
                'execution': {
                    'remove_unnecessary_outputs': False,
                    'stop_on_first_crash': True,
                    'stop_on_first_rerun': False,
                    'crashfile_format': "txt",
                    'use_relative_paths': True,
                    'write_provenance': False
                }
            }
        )

        # Update nypipe logging with config
        nipype_logging.update_logging(config)
        # config.enable_provenance()

        if self.use_manual_masks:
            # R. From what I get, the nipype datagrabber has two output fields
            # because it's loading both the T2 images and the masks available,
            # rather than computing them. 
            dg = pe.Node(
                interface=DataGrabber(outfields=['T2ws', 'masks']),
                name='data_grabber'
            )
            dg.inputs.base_directory = self.bids_dir
            dg.inputs.template = '*'
            dg.inputs.raise_on_empty = False
            dg.inputs.sort_filelist = True

            if self.session is not None:
                t2ws_template = os.path.join(
                    self.subject, self.session, 'anat',
                    '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])
                )
                if self.m_masks_desc is not None:
                    masks_template = os.path.join(
                        'derivatives', self.m_masks_derivatives_dir, self.subject, self.session, 'anat',
                        '_'.join([sub_ses, '*_run-*', '_desc-'+self.m_masks_desc, '*mask.nii.gz'])
                    )
                else:
                    masks_template = os.path.join(
                        'derivatives', self.m_masks_derivatives_dir, self.subject, self.session, 'anat',
                        '_'.join([sub_ses, '*run-*', '*mask.nii.gz'])
                    )
            else:
                t2ws_template=os.path.join(self.subject, 'anat', sub_ses + '*_run-*_T2w.nii.gz')
                # Q. It seems that this is never used: it's in the branch
                # self.session = None and it passes it as argument to
                # os.path.join()
                if self.m_masks_desc is not None:
                    masks_template = os.path.join(
                        'derivatives', self.m_masks_derivatives_dir, self.subject, self.session, 'anat',
                        '_'.join([sub_ses, '*_run-*', '_desc-'+self.m_masks_desc, '*mask.nii.gz'])
                    )
                else:
                    masks_template = os.path.join(
                        'derivatives', self.m_masks_derivatives_dir, self.subject, 'anat',
                        sub_ses + '*_run-*_*mask.nii.gz'
                    )

            dg.inputs.field_template = dict(T2ws=t2ws_template,
                                            masks=masks_template)

            brainMask = pe.MapNode(interface=IdentityInterface(fields=['out_file']),
                                name='brain_masks_bypass',
                                iterfield=['out_file'])

            if self.m_stacks is not None:
                custom_masks_filter = pe.Node(interface=preprocess.FilteringByRunid(),
                                           name='custom_masks_filter')
                custom_masks_filter.inputs.stacks_id = self.m_stacks

        else:
            # R. Only grabbing the T2 images, masks to be computed
            dg = pe.Node(interface=DataGrabber(outfields=['T2ws']),
                      name='data_grabber')

            dg.inputs.base_directory = self.bids_dir
            dg.inputs.template = '*'
            dg.inputs.raise_on_empty = False
            dg.inputs.sort_filelist = True

            dg.inputs.field_template = dict(T2ws=os.path.join(self.subject,
                                                              'anat',
                                                              sub_ses+'*_run-*_T2w.nii.gz'))
            if self.session is not None:
                dg.inputs.field_template = dict(T2ws=os.path.join(self.subject,
                                                                  self.session, 'anat', '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])))

            if self.m_stacks is not None:
                t2ws_filter_prior_masks = pe.Node(interface=preprocess.FilteringByRunid(),
                                               name='t2ws_filter_prior_masks')
                t2ws_filter_prior_masks.inputs.stacks_id = self.m_stacks
            # Computing brain masks
            brainMask = pe.MapNode(interface = preprocess.BrainExtraction(),
                                name='brainExtraction',
                                iterfield=['in_file'])

            # I think that this is describing the path to first the
            # localization model, and then to the thresholding model. 
            # Q. Are the thresholds for localization and segmentation chosen
            # arbitrarily? How does it look with various values?
            brainMask.inputs.in_ckpt_loc = pkg_resources.resource_filename(
                "pymialsrtk",
                os.path.join("data",
                             "Network_checkpoints",
                             "Network_checkpoints_localization",
                             "Unet.ckpt-88000.index")
            ).split('.index')[0]
            brainMask.inputs.threshold_loc = 0.49
            brainMask.inputs.in_ckpt_seg = pkg_resources.resource_filename(
                "pymialsrtk",
                os.path.join("data",
                             "Network_checkpoints",
                             "Network_checkpoints_segmentation",
                             "Unet.ckpt-20000.index")
            ).split('.index')[0]
            brainMask.inputs.threshold_seg = 0.5

        t2ws_filtered = pe.Node(interface=preprocess.FilteringByRunid(),
                             name='t2ws_filtered')
        masks_filtered = pe.Node(interface=preprocess.FilteringByRunid(),
                              name='masks_filtered')

        if not self.m_skip_stacks_ordering:
            stacksOrdering = pe.Node(interface=preprocess.StacksOrdering(),
                                  name='stackOrdering')
        else:
            stacksOrdering = pe.Node(interface=IdentityInterface(fields=['stacks_order']),
                                  name='stackOrdering')
            stacksOrdering.inputs.stacks_order = self.m_stacks

        preprocessing_stage = preproc_stage.create_preproc_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising)

        registration_stage = create_registration_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_skip_svr=self.m_skip_svr,
            p_sub_ses=sub_ses)

        srtkMaskImage01 = pe.MapNode(interface=preprocess.MialsrtkMaskImage(),
                                  name='srtkMaskImage01',
                                  iterfield=['in_file', 'in_mask'])

        if self.m_do_nlm_denoising:
            srtkMaskImage01_nlm = pe.MapNode(
                interface=preprocess.MialsrtkMaskImage(),
                name='srtkMaskImage01_nlm',
                iterfield=['in_file', 'in_mask'])


        srtkN4BiasFieldCorrection = pe.Node(interface=postprocess.MialsrtkN4BiasFieldCorrection(),
                                         name='srtkN4BiasFieldCorrection')


        srtkMaskImage02 = pe.Node(interface=preprocess.MialsrtkMaskImage(),
                               name='srtkMaskImage02')

        # Q. How does the output_mgmt_stage work? It seems that it does collect
        # the outputs and has the data sinker, but can you connect anything to it?

        prepro_mgmt_stage = create_prepro_output_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            name='prepro_mgmt_stage')

        prepro_mgmt_stage.inputs.inputnode.sub_ses = sub_ses
        prepro_mgmt_stage.inputs.inputnode.sr_id = self.sr_id
        prepro_mgmt_stage.inputs.inputnode.use_manual_masks = self.use_manual_masks
        prepro_mgmt_stage.inputs.inputnode.final_res_dir = final_res_dir

        # Build workflow : connections of the nodes
        # Nodes ready : Linking now
        if self.use_manual_masks:
            if self.m_stacks is not None:
                self.wf.connect(dg, "masks", custom_masks_filter, "input_files")
                self.wf.connect(custom_masks_filter, "output_files", brainMask, "out_file")
            else:
                self.wf.connect(dg, "masks", brainMask, "out_file")
        else:
            # R. Recall that m_stacks describes a custom set of stacks given
            # for the reconstruction
            if self.m_stacks is not None:
                self.wf.connect(dg, "T2ws", t2ws_filter_prior_masks, "input_files")
                self.wf.connect(t2ws_filter_prior_masks, "output_files", brainMask, "in_file")
            else:
                self.wf.connect(dg, "T2ws", brainMask, "in_file")

        if not self.m_skip_stacks_ordering:
            self.wf.connect(brainMask, "out_file", stacksOrdering, "input_masks")

        self.wf.connect(stacksOrdering, "stacks_order", t2ws_filtered, "stacks_id")
        self.wf.connect(dg, "T2ws", t2ws_filtered, "input_files")

        self.wf.connect(stacksOrdering, "stacks_order", masks_filtered, "stacks_id")
        self.wf.connect(brainMask, "out_file", masks_filtered, "input_files")

        self.wf.connect(t2ws_filtered, "output_files",
                        preprocessing_stage, "inputnode.input_images")
        self.wf.connect(masks_filtered, "output_files",
                        preprocessing_stage, "inputnode.input_masks")

        self.wf.connect(preprocessing_stage, ("outputnode.output_masks", utils.sort_ascending),
                        srtkMaskImage01, "in_mask")

        self.wf.connect(preprocessing_stage, ("outputnode.output_images", utils.sort_ascending),
                        srtkMaskImage01, "in_file")

        if self.m_do_nlm_denoising:
            self.wf.connect(preprocessing_stage, ("outputnode.output_images_nlm",
                                                  utils.sort_ascending),
                            srtkMaskImage01_nlm, "in_file")
            self.wf.connect(preprocessing_stage, ("outputnode.output_masks", utils.sort_ascending),
                            srtkMaskImage01_nlm, "in_mask")
            self.wf.connect(srtkMaskImage01_nlm, ("out_im_file",
                                                  utils.sort_ascending),
                            registration_stage, "inputnode.input_images_nlm")

        self.wf.connect(srtkMaskImage01, ("out_im_file", utils.sort_ascending),
                        registration_stage, "inputnode.input_images")

        self.wf.connect(preprocessing_stage, "outputnode.output_masks",
                        registration_stage, "inputnode.input_masks")

        self.wf.connect(stacksOrdering, "stacks_order",
                        registration_stage, "inputnode.stacks_order")

        self.wf.connect(stacksOrdering, "stacks_order", prepro_mgmt_stage, "inputnode.stacks_order")

        self.wf.connect(preprocessing_stage, "outputnode.output_masks",
                        prepro_mgmt_stage, "inputnode.input_masks")
        #self.wf.connect(preprocessing_stage, "outputnode.output_images",
        #                prepro_mgmt_stage, "inputnode.input_images")

        self.wf.connect(registration_stage, "outputnode.output_sdi",
                        prepro_mgmt_stage, "inputnode.input_sdi")
        self.wf.connect(registration_stage, "outputnode.output_transforms",
                        prepro_mgmt_stage, "inputnode.input_transforms")

        if self.m_do_nlm_denoising:
            self.wf.connect(srtkMaskImage01_nlm, "out_im_file",
                            prepro_mgmt_stage, "inputnode.input_images_nlm")
        
        if not self.m_skip_stacks_ordering:
            self.wf.connect(stacksOrdering, "report_image",
                            prepro_mgmt_stage, "inputnode.report_image")
            self.wf.connect(stacksOrdering, "motion_tsv",
                            prepro_mgmt_stage, "inputnode.motion_tsv")

    def run(self, memory=None):
        """Execute the workflow of preprocessing pipeline.

        Nipype execution engine will take care of the management and execution of
        all processing steps involved in the super-resolution reconstruction pipeline.
        Note that the complete execution graph is saved as a PNG image to support
        transparency on the whole processing.

        Parameters
        ----------
        memory : int
            Maximal memory used by the workflow
        """

        # Use nipype.interface logger to print some information messages
        iflogger = nipype_logging.getLogger('nipype.interface')
        iflogger.info("**** Workflow graph creation ****")
        self.wf.write_graph(dotfilename='graph.dot', graph2use='colored', format='png', simple_form=True)

        # Copy and rename the generated "graph.png" image
        src = os.path.join(self.wf.base_dir, self.wf.name, 'graph.png')
        dst_base = os.path.join(self.output_dir,
        '-'.join(["pymialsrtk", __version__]),
        self.subject,
        )
        print("session: ",self.session)
        if self.session is not None:
            dst_base = os.path.join(dst_base, self.session)
        if self.session is not None:
            dst = os.path.join(
                self.output_dir,
                '-'.join(["pymialsrtk", __version__]),
                self.subject,
                self.session,
                'figures',
                f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
            )
            dst2 = os.path.join(dst_base, 
                    'figures',
                    f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
                    )
            print(dst == dst2, dst2)
        else:
            dst = os.path.join(
                    self.output_dir,
                    '-'.join(["pymialsrtk", __version__]),
                    self.subject,
                    'figures',
                    f'{self.subject}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
            )
            dst2 = os.path.join(dst_base, 
                    'figures',
                    f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
                    )
            print(dst == dst2, dst2)
        # Create the figures/ and parent directories if they do not exist
        figures_dir = os.path.dirname(dst)
        os.makedirs(figures_dir, exist_ok=True)
        # Make the copy
        iflogger.info(f'\t > Copy {src} to {dst}...')
        shutil.copy(src=src, dst=dst)

        # Create dictionary of arguments passed to plugin_args
        args_dict = {
            'raise_insufficient': False,
            'n_procs': self.nipype_number_of_cores
        }

        if (memory is not None) and (memory > 0):
            args_dict['memory_gb'] = memory

        iflogger.info("**** Processing ****")
        # datetime object containing current start date and time
        start = datetime.now()
        self.run_start_time = start.strftime("%B %d, %Y / %H:%M:%S")
        print(f" Start date / time : {self.run_start_time}")

        # Execute the workflow
        #if self.nipype_number_of_cores > 1:
        #    res = self.wf.run(plugin='MultiProc', plugin_args=args_dict)
        #else:
        #    res = self.wf.run()

        # Copy and rename the workflow execution log
        src = os.path.join(self.wf.base_dir, "pypeline.log")
        if self.session is not None:
            dst = os.path.join(
                    self.output_dir,
                    '-'.join(["pymialsrtk", __version__]),
                    self.subject,
                    self.session,
                    'logs',
                    f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_log.txt'
            )           
            dst2 = os.path.join(dst_base, 
                    'logs',
                    f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_log.txt'
                    )
            print(dst == dst2, dst2)
        else:
            dst = os.path.join(
                    self.output_dir,
                    '-'.join(["pymialsrtk", __version__]),
                    self.subject,
                    'logs',
                    f'{self.subject}_rec-SR_id-{self.sr_id}_log.txt'
            )
            dst2 = os.path.join(dst_base, 
                    'logs',
                    f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_log.txt'
                    )
            print(dst == dst2, dst2)
        # Create the logs/ and parent directories if they do not exist
        logs_dir = os.path.dirname(dst)
        os.makedirs(logs_dir, exist_ok=True)
        # Make the copy
        iflogger.info(f'\t > Copy {src} to {dst}...')
        shutil.copy(src=src, dst=dst)

        # datetime object containing current end date and time
        end = datetime.now()
        self.run_end_time = end.strftime("%B %d, %Y / %H:%M:%S")
        print(f" End date / time : {self.run_end_time}")

        # Compute elapsed running time in minutes and seconds
        duration = end - start
        (minutes, seconds) = divmod(duration.total_seconds(), 60)
        self.run_elapsed_time = f'{int(minutes)} minutes and {int(seconds)} seconds'
        print(f" Elapsed time: {self.run_end_time}")

        iflogger.info("**** Write dataset derivatives description ****")
        for toolbox in ["pymialsrtk", "nipype"]:
            write_bids_derivative_description(
                bids_dir=self.bids_dir,
                deriv_dir=self.output_dir,
                pipeline_name=toolbox
            )

        iflogger.info("**** Run complete ****")

        return res
