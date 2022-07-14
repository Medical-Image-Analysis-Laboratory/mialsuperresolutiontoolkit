# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the super-resolution reconstruction pipeline."""

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
from nipype.pipeline import Node, MapNode, Workflow
from nipype.interfaces.utility import IdentityInterface

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.workflows.preproc_stage as preproc_stage
import pymialsrtk.workflows.recon_stage as recon_stage
import pymialsrtk.workflows.recon_labels_stage as recon_labels_stage
import pymialsrtk.interfaces.utils as utils
from pymialsrtk.bids.utils import write_bids_derivative_description

# Get pymialsrtk version
from pymialsrtk.info import __version__


class AnatomicalPipeline:
    """Class used to represent the workflow of the Super-Resolution reconstruction pipeline.

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

    pipeline_name = "srr_pipeline"
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
            self.m_do_reconstruct_labels = p_dict_custom_interfaces['do_reconstruct_labels'] if 'do_reconstruct_labels' in p_dict_custom_interfaces.keys() else False

            self.m_skip_stacks_ordering = p_dict_custom_interfaces['skip_stacks_ordering'] if \
                ((self.m_stacks is not None) and ('skip_stacks_ordering' in p_dict_custom_interfaces.keys())) else False
        else:
            self.m_skip_svr = False
            self.m_do_refine_hr_mask = False
            self.m_do_nlm_denoising =  False
            self.m_skip_stacks_ordering = False
            self.m_do_reconstruct_labels = False

    def create_workflow(self):
        """Create the Niype workflow of the super-resolution pipeline.

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
        if os.path.isfile(os.path.join(wf_base_dir, "pypeline.log")):
            os.unlink(os.path.join(wf_base_dir, "pypeline.log"))

        self.wf = Workflow(name=self.pipeline_name,base_dir=wf_base_dir)

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

            dg = Node(
                interface=DataGrabber(outfields=(['T2ws', 'masks', 'labels'] if self.m_do_reconstruct_labels else ['T2ws', 'masks'])),
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
                if self.m_do_reconstruct_labels:
                    labels_template = os.path.join(
                        'derivatives', 'labels', self.subject, self.session, 'anat',
                        '_'.join([sub_ses, '*run-*', '*labels.nii.gz'])
                    )
            else:
                t2ws_template=os.path.join(self.subject, 'anat', sub_ses + '*_run-*_T2w.nii.gz')

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

                if self.m_do_reconstruct_labels:
                    labels_template = os.path.join(
                        'derivatives', 'labels', self.subject, 'anat',
                        '_'.join([sub_ses, '*run-*', '*labels.nii.gz'])
                    )


            if self.m_do_reconstruct_labels:
                dg.inputs.field_template = dict(T2ws=t2ws_template,
                                            masks=masks_template,
                                            labels=labels_template)
            else:
                dg.inputs.field_template = dict(T2ws=t2ws_template,
                                                masks=masks_template)

            brainMask = MapNode(interface=IdentityInterface(fields=['out_file']),
                                name='brain_masks_bypass',
                                iterfield=['out_file'])

            if self.m_stacks is not None:
                custom_masks_filter = Node(interface=preprocess.FilteringByRunid(),
                                           name='custom_masks_filter')
                custom_masks_filter.inputs.stacks_id = self.m_stacks

        else:
            dg = Node(interface=DataGrabber(outfields=['T2ws']),
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
                t2ws_filter_prior_masks = Node(interface=preprocess.FilteringByRunid(),
                                               name='t2ws_filter_prior_masks')
                t2ws_filter_prior_masks.inputs.stacks_id = self.m_stacks

            brainMask = MapNode(interface = preprocess.BrainExtraction(),
                                name='brainExtraction',
                                iterfield=['in_file'])

            brainMask.inputs.bids_dir = self.bids_dir
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

        t2ws_filtered = Node(interface=preprocess.FilteringByRunid(),
                             name='t2ws_filtered')
        masks_filtered = Node(interface=preprocess.FilteringByRunid(),
                              name='masks_filtered')

        if self.m_do_reconstruct_labels:
            labels_filtered = Node(interface=preprocess.FilteringByRunid(),
                              name='labels_filtered')

            reconstruct_labels_stage = recon_labels_stage.create_recon_labels_stage(sub_ses=sub_ses)
            reconstruct_labels_stage.inputs.inputnode.label_ids = [0,1,2,3,4,5,6,7]

        if not self.m_skip_stacks_ordering:
            stacksOrdering = Node(interface=preprocess.StacksOrdering(),
                                  name='stackOrdering')
        else:
            stacksOrdering = Node(interface=IdentityInterface(fields=['stacks_order']),
                                  name='stackOrdering')
            stacksOrdering.inputs.stacks_order = self.m_stacks

        preprocessing_stage = preproc_stage.create_preproc_stage(p_do_nlm_denoising=self.m_do_nlm_denoising,
                                                                 p_do_reconstruct_labels=self.m_do_reconstruct_labels,
                                                                 bids_dir=self.bids_dir)

        reconstruction_stage = recon_stage.create_recon_stage(
            p_paramTV=self.paramTV,
            p_use_manual_masks=self.use_manual_masks,
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_do_refine_hr_mask=self.m_do_refine_hr_mask,
            p_skip_svr=self.m_skip_svr,
            p_sub_ses=sub_ses,
            p_bids_dir=self.bids_dir)

        srtkMaskImage01 = MapNode(interface=preprocess.MialsrtkMaskImage(),
                                  name='srtkMaskImage01',
                                  iterfield=['in_file', 'in_mask'])
        srtkMaskImage01.inputs.bids_dir = self.bids_dir
        if self.m_do_nlm_denoising:
            srtkMaskImage01_nlm = MapNode(
                interface=preprocess.MialsrtkMaskImage(),
                name='srtkMaskImage01_nlm',
                iterfield=['in_file', 'in_mask'])
            srtkMaskImage01_nlm.inputs.bids_dir = self.bids_dir

        srtkN4BiasFieldCorrection = Node(interface=postprocess.MialsrtkN4BiasFieldCorrection(),
                                         name='srtkN4BiasFieldCorrection')
        srtkN4BiasFieldCorrection.inputs.bids_dir = self.bids_dir

        srtkMaskImage02 = Node(interface=preprocess.MialsrtkMaskImage(),
                               name='srtkMaskImage02')
        srtkMaskImage02.inputs.bids_dir = self.bids_dir

        # Build workflow : connections of the nodes
        # Nodes ready : Linking now
        if self.use_manual_masks:
            if self.m_stacks is not None:
                self.wf.connect(dg, "masks", custom_masks_filter, "input_files")
                self.wf.connect(custom_masks_filter, "output_files", brainMask, "out_file")
            else:
                self.wf.connect(dg, "masks", brainMask, "out_file")
        else:
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
                            reconstruction_stage, "inputnode.input_images_nlm")

        self.wf.connect(srtkMaskImage01, ("out_im_file", utils.sort_ascending),
                        reconstruction_stage, "inputnode.input_images")

        self.wf.connect(preprocessing_stage, "outputnode.output_masks",
                        reconstruction_stage, "inputnode.input_masks")

        self.wf.connect(stacksOrdering, "stacks_order",
                        reconstruction_stage, "inputnode.stacks_order")

        self.wf.connect(reconstruction_stage, "outputnode.output_sr",
                        srtkMaskImage02, "in_file")
        self.wf.connect(reconstruction_stage, "outputnode.output_hr_mask",
                        srtkMaskImage02, "in_mask")

        self.wf.connect(reconstruction_stage, "outputnode.output_sr",
                        srtkN4BiasFieldCorrection, "input_image")
        self.wf.connect(srtkMaskImage02, "out_im_file",
                        srtkN4BiasFieldCorrection, "input_mask")

        if self.m_do_reconstruct_labels:
            self.wf.connect(stacksOrdering, "stacks_order", labels_filtered, "stacks_id")
            self.wf.connect(dg, "labels", labels_filtered, "input_files")
            self.wf.connect(labels_filtered, "output_files",
                        preprocessing_stage, "inputnode.input_labels")

            self.wf.connect(preprocessing_stage, "outputnode.output_labels",
                            reconstruct_labels_stage, "inputnode.input_labels")
            self.wf.connect(preprocessing_stage, "outputnode.output_masks",
                            reconstruct_labels_stage, "inputnode.input_masks")
            self.wf.connect(reconstruction_stage, "outputnode.output_transforms",
                            reconstruct_labels_stage, "inputnode.input_transforms")

            self.wf.connect(reconstruction_stage, "outputnode.output_sdi",
                            reconstruct_labels_stage, "inputnode.input_reference")
            self.wf.connect(stacksOrdering, "stacks_order",
                            reconstruct_labels_stage, "inputnode.stacks_order")

        # Datasinker
        finalFilenamesGeneration = Node(interface=postprocess.FilenamesGeneration(),
                                        name='filenames_gen')
        finalFilenamesGeneration.inputs.sub_ses = sub_ses
        finalFilenamesGeneration.inputs.sr_id = self.sr_id
        finalFilenamesGeneration.inputs.use_manual_masks = self.use_manual_masks

        self.wf.connect(stacksOrdering, "stacks_order", finalFilenamesGeneration, "stacks_order")

        datasink = Node(interface=DataSink(), name='data_sinker')
        datasink.inputs.base_directory = final_res_dir
#
        if not self.m_skip_stacks_ordering:
            self.wf.connect(stacksOrdering, "report_image",
                            datasink, 'figures.@stackOrderingQC')
            self.wf.connect(stacksOrdering, "motion_tsv",
                            datasink, 'anat.@motionTSV')
        self.wf.connect(preprocessing_stage, "outputnode.output_masks",
                        datasink, 'anat.@LRmasks')
        self.wf.connect(preprocessing_stage, "outputnode.output_images",
                        datasink, 'anat.@LRsPreproc')
        self.wf.connect(reconstruction_stage, "outputnode.output_transforms",
                        datasink, 'xfm.@transforms')
        self.wf.connect(finalFilenamesGeneration, "substitutions",
                        datasink, "substitutions")
        self.wf.connect(srtkMaskImage01, "out_im_file",
                        datasink, 'anat.@LRsDenoised')
        self.wf.connect(reconstruction_stage, "outputnode.output_sdi",
                        datasink, 'anat.@SDI')
        self.wf.connect(srtkN4BiasFieldCorrection, "output_image",
                        datasink, 'anat.@SR')
        self.wf.connect(reconstruction_stage, "outputnode.output_json_path",
                        datasink, 'anat.@SRjson')
        self.wf.connect(reconstruction_stage, "outputnode.output_sr_png",
                        datasink, 'figures.@SRpng')
        self.wf.connect(reconstruction_stage, "outputnode.output_hr_mask",
                        datasink, 'anat.@SRmask')
#
        if self.m_do_reconstruct_labels:
            self.wf.connect(reconstruct_labels_stage, "outputnode.output_labelmap",
                            datasink, 'anat.@HRlabel')


    def run(self, memory=None):
        """Execute the workflow of the super-resolution reconstruction pipeline.

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
        if self.session is not None:
            dst = os.path.join(
                self.output_dir,
                '-'.join(["pymialsrtk", __version__]),
                self.subject,
                self.session,
                'figures',
                f'{self.subject}_{self.session}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
            )
        else:
            dst = os.path.join(
                    self.output_dir,
                    '-'.join(["pymialsrtk", __version__]),
                    self.subject,
                    'figures',
                    f'{self.subject}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
            )
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
        if self.nipype_number_of_cores > 1:
            res = self.wf.run(plugin='MultiProc', plugin_args=args_dict)
        else:
            res = self.wf.run()

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
        else:
            dst = os.path.join(
                    self.output_dir,
                    '-'.join(["pymialsrtk", __version__]),
                    self.subject,
                    'logs',
                    f'{self.subject}_rec-SR_id-{self.sr_id}_log.txt'
            )
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

        iflogger.info("**** Super-resolution HTML report creation ****")
        self.create_subject_report()

        return res

    def create_subject_report(self):
        """Create the HTML report"""
        # Set main subject derivatives directory
        if self.session is None:
            sub_ses = self.subject
            final_res_dir = os.path.join(self.output_dir,
                                         '-'.join(["pymialsrtk", __version__]),
                                         self.subject)
        else:
            sub_ses = f'{self.subject}_{self.session}'
            final_res_dir = os.path.join(self.output_dir,
                                         '-'.join(["pymialsrtk", __version__]),
                                         self.subject,
                                         self.session)
        # Get the HTML report template
        path = pkg_resources.resource_filename(
            'pymialsrtk',
            "data/report/templates/template.html"
        )
        jinja_template_dir = os.path.dirname(path)

        file_loader = FileSystemLoader(jinja_template_dir)
        env = Environment(loader=file_loader)

        template = env.get_template('template.html')

        # Load main data derivatives necessary for the report
        sr_nii_image = os.path.join(
            final_res_dir, 'anat',
            f'{sub_ses}_rec-SR_id-{self.sr_id}_T2w.nii.gz'
        )
        img = nib.load(sr_nii_image)
        sx, sy, sz = img.header.get_zooms()

        sr_json_metadata = os.path.join(
            final_res_dir, 'anat',
            f'{sub_ses}_rec-SR_id-{self.sr_id}_T2w.json'
        )
        with open(sr_json_metadata) as f:
            sr_json_metadata = json.load(f)

        workflow_image = os.path.join(
            '..', 'figures',
            f'{sub_ses}_rec-SR_id-{self.sr_id}_desc-processing_graph.png'
        )

        sr_png_image = os.path.join(
            '..', 'figures',
            f'{sub_ses}_rec-SR_id-{self.sr_id}_T2w.png'
        )

        motion_report_image = os.path.join(
            '..', 'figures',
            f'{sub_ses}_rec-SR_id-{self.sr_id}_desc-motion_stats.png'
        )

        log_file = os.path.join(
            '..', 'logs',
            f'{sub_ses}_rec-SR_id-{self.sr_id}_log.txt'
        )

        # Create the text for {{subject}} and {{session}} fields in template
        report_subject_text = f'{self.subject.split("-")[-1]}'
        if self.session is not None:
            report_session_text = f'{self.session.split("-")[-1]}'
        else:
            report_session_text = None

        # Generate the report
        report_html_content = template.render(
            subject=report_subject_text,
            session=report_session_text,
            processing_datetime=self.run_start_time,
            run_time=self.run_elapsed_time,
            log=log_file,
            sr_id=self.sr_id,
            stacks=self.m_stacks,
            svr="on" if not self.m_skip_svr else "off",
            nlm_denoising="on" if self.m_do_nlm_denoising else "off",
            stacks_ordering="on" if not self.m_skip_stacks_ordering else "off",
            do_refine_hr_mask="on" if self.m_do_refine_hr_mask else "off",
            use_auto_masks="on" if self.m_masks_derivatives_dir is None else "off",
            custom_masks_dir=self.m_masks_derivatives_dir if self.m_masks_derivatives_dir is not None else None,
            sr_resolution=f"{sx} x {sy} x {sz} mm<sup>3</sup>",
            sr_json_metadata=sr_json_metadata,
            workflow_graph=workflow_image,
            sr_png_image=sr_png_image,
            motion_report_image=motion_report_image,
            version=__version__,
            os=f'{platform.system()} {platform.release()}',
            python=f'{sys.version}',
            openmp_threads=self.openmp_number_of_cores,
            nipype_threads=self.nipype_number_of_cores,
            jinja_version=__jinja2_version__
        )
        # Create the report directory if it does not exist
        report_dir = os.path.join(final_res_dir, 'report')
        os.makedirs(report_dir, exist_ok=True)

        # Save the HTML report file
        out_report_filename = os.path.join(report_dir, f'{sub_ses}.html')
        print(f'\t* Save HTML report as {out_report_filename}...')
        with open(out_report_filename, "w+") as file:
            file.write(report_html_content)
