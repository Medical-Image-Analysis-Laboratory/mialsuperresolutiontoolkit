# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the preprocessing pipeline."""

import os

from nipype.info import __version__ as __nipype_version__
from nipype import config
from nipype import logging as nipype_logging
from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.utils as utils

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.reconstruction as reconstruction
from pymialsrtk.workflows.input_stage import create_input_stage
import pymialsrtk.workflows.preproc_stage as preproc_stage
from pymialsrtk.workflows.output_stage import create_preproc_output_stage
from pymialsrtk.workflows.registration_stage import create_registration_stage
from .abstract import AbstractAnatomicalPipeline

# Get pymialsrtk version
from pymialsrtk.info import __version__


class PreprocessingPipeline(AbstractAnatomicalPipeline):
    """Class used to represent the workflow of the
    Preprocessing pipeline.

    Attributes
    -----------
    m_bids_dir : string
        BIDS root directory (required)

    m_output_dir : string
        Output derivatives directory (required)

    m_subject : string
        Subject ID (in the form ``sub-XX``)

    m_wf : nipype.pipeline.Workflow
        Nipype workflow of the preprocessing pipeline

    m_sr_id : string
        ID of the preprocessing useful to distinguish when multiple
        preprocessing with different order of stacks are run on
        the same subject

    m_session : string
        Session ID if applicable (in the form ``ses-YY``)

    m_stacks : list(int)
        List of stack to be used in the preprocessing.
        The specified order is kept if `skip_stacks_ordering` is True.

    m_masks_derivatives_dir : string
        directory basename in BIDS directory derivatives where to search
        for masks (optional)

    m_do_nlm_denoising : bool
        Weither the NLM denoising preprocessing should be performed prior to
        motion estimation. (default is False)

    m_skip_stacks_ordering : bool (optional)
        Weither the automatic stacks ordering should be skipped.
        (default is False)

    Examples
    --------
    >>> from pymialsrtk.pipelines.anatomical.srr import PreprocessingPipeline
    >>> # Create a new instance
    >>> pipeline = PreprocessingPipeline(bids_dir='/path/to/bids_dir',
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
    m_pipeline_name = "preproc_pipeline"

    # Custom interfaces options
    m_do_registration = None
    m_skip_svr = None

    def __init__(
        self,
        p_bids_dir,
        p_output_dir, p_subject,
        p_ga=None,
        p_stacks=None,
        sr_id=1,
        p_session=None,
        p_masks_derivatives_dir=None,
        p_masks_desc=None,
        p_dict_custom_interfaces=None,
        p_openmp_number_of_cores=None,
        p_nipype_number_of_cores=None,
    ):
        """Constructor of PreprocessingPipeline class instance."""

        super().__init__(p_bids_dir, p_output_dir, p_subject, p_ga, p_stacks,
                         sr_id, p_session, p_masks_derivatives_dir,
                         p_masks_desc, p_dict_custom_interfaces,
                         p_openmp_number_of_cores, p_nipype_number_of_cores,
                         "pre"
                         )

        if p_dict_custom_interfaces is not None:
            self.m_do_registration = \
                p_dict_custom_interfaces['prepro_do_registration'] \
                if 'prepro_do_registration' in p_dict_custom_interfaces.keys()\
                else False
            self.m_skip_svr = p_dict_custom_interfaces['skip_svr'] \
                if 'skip_svr' in p_dict_custom_interfaces.keys() \
                else False
            if 'do_reconstruct_labels' in p_dict_custom_interfaces.keys() and \
               p_dict_custom_interfaces['do_reconstruct_labels']:
                raise RuntimeError("do_reconstruct_labels should not be "
                                   "enabled when run_type=preprocessing.")
        else:
            self.m_do_registration = False
            self.m_skip_svr = False

    def create_workflow(self):
        """Create the Niype workflow of the super-resolution pipeline.

        It is composed of a succession of Nodes and their corresponding
        parameters, where the output of node i goes to the input of node i+1.

        """

        self.m_wf = pe.Workflow(name=self.m_pipeline_name,
                                base_dir=self.m_wf_base_dir
                                )

        config.update_config(
            {
                'logging': {
                      'log_directory': os.path.join(self.m_wf_base_dir),
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

        input_stage = create_input_stage(
            self.m_bids_dir,
            self.m_subject,
            self.m_session,
            self.m_use_manual_masks,
            self.m_masks_desc,
            self.m_masks_derivatives_dir,
            None,
            self.m_skip_stacks_ordering,
            False,
            self.m_stacks
        )

        preprocessing_stage = preproc_stage.create_preproc_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising)

        prepro_mgmt_stage = create_preproc_output_stage(
                    p_do_nlm_denoising=self.m_do_nlm_denoising,
                    p_do_registration=self.m_do_registration,
                    name='prepro_mgmt_stage')

        prepro_mgmt_stage.inputs.inputnode.sub_ses = self.m_sub_ses
        prepro_mgmt_stage.inputs.inputnode.sr_id = self.m_sr_id
        prepro_mgmt_stage.inputs.inputnode.use_manual_masks = \
            self.m_use_manual_masks
        prepro_mgmt_stage.inputs.inputnode.final_res_dir = self.m_final_res_dir
        prepro_mgmt_stage.inputs.inputnode.run_type = self.run_type

        if self.m_do_registration:
            registration_stage = create_registration_stage(
                p_do_nlm_denoising=self.m_do_nlm_denoising,
                p_skip_svr=self.m_skip_svr,
                p_sub_ses=self.m_sub_ses)

        # Build workflow : connections of the nodes
        # Nodes ready : Linking now
        self.m_wf.connect(input_stage, "outputnode.t2ws_filtered",
                          preprocessing_stage, "inputnode.input_images")

        self.m_wf.connect(input_stage, "outputnode.masks_filtered",
                          preprocessing_stage, "inputnode.input_masks")

        if self.m_do_registration:
            if self.m_do_nlm_denoising:
                self.m_wf.connect(preprocessing_stage,
                                  ("outputnode.output_images_nlm",
                                   utils.sort_ascending
                                   ),
                                  registration_stage,
                                  "inputnode.input_images_nlm"
                                  )

            self.m_wf.connect(preprocessing_stage, ("outputnode.output_images",
                                                    utils.sort_ascending),
                              registration_stage, "inputnode.input_images")

            self.m_wf.connect(preprocessing_stage, ("outputnode.output_masks",
                                                    utils.sort_ascending),
                              registration_stage, "inputnode.input_masks")

            self.m_wf.connect(input_stage, "outputnode.stacks_order",
                              registration_stage, "inputnode.stacks_order")

            self.m_wf.connect(registration_stage, "outputnode.output_sdi",
                              prepro_mgmt_stage, "inputnode.input_sdi")

            self.m_wf.connect(registration_stage,
                              "outputnode.output_transforms",
                              prepro_mgmt_stage,
                              "inputnode.input_transforms")
        else:
            self.m_wf.connect(preprocessing_stage, ("outputnode.output_images",
                                                    utils.sort_ascending),
                              prepro_mgmt_stage, "inputnode.input_images")

        self.m_wf.connect(input_stage, "outputnode.stacks_order",
                          prepro_mgmt_stage, "inputnode.stacks_order")

        self.m_wf.connect(preprocessing_stage, "outputnode.output_masks",
                          prepro_mgmt_stage, "inputnode.input_masks")

        if self.m_do_nlm_denoising:
            self.m_wf.connect(preprocessing_stage,
                              ("outputnode.output_images_nlm",
                               utils.sort_ascending),
                              prepro_mgmt_stage, "inputnode.input_images_nlm"
                              )

        if not self.m_skip_stacks_ordering:
            self.m_wf.connect(input_stage, "outputnode.report_image",
                              prepro_mgmt_stage, "inputnode.report_image")
            self.m_wf.connect(input_stage, "outputnode.motion_tsv",
                              prepro_mgmt_stage, "inputnode.motion_tsv")
