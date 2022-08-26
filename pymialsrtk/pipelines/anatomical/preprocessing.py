# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the preprocessing pipeline."""

import os

from jinja2 import Environment, FileSystemLoader
from jinja2 import __version__ as __jinja2_version__

from nipype.info import __version__ as __nipype_version__
from nipype import config
from nipype import logging as nipype_logging

from nipype.pipeline import engine as pe

import pymialsrtk.interfaces.utils as utils

# Import the implemented interface from pymialsrtk
from pymialsrtk.workflows.input_stage import create_input_stage
import pymialsrtk.workflows.preproc_stage as preproc_stage
import pymialsrtk.workflows.recon_stage as recon_stage
import pymialsrtk.workflows.postproc_stage as postproc_stage
import pymialsrtk.workflows.srr_output_stage as srr_output_stage
from anatomical_pipeline import AnatomicalPipeline

# Get pymialsrtk version
from pymialsrtk.info import __version__


class PreprocessingPipeline(AnatomicalPipeline):
    """Class used to represent the workflow of the preprocessing
    pipeline.

    Attributes
    -----------
    bids_dir : string
        BIDS root directory (required)

    output_dir : string
        Output derivatives directory (required)

    subject : string
        Subject ID (in the form ``sub-XX``)

    wf : nipype.pipeline.Workflow
        Nipype workflow of the preprocessing pipeline

    deltatTV : string
        Super-resolution optimization time-step

    lambdaTV : float
        Regularization weight (default is 0.75)

    num_iterations : string
        Number of iterations in the primal/dual loops used in the optimization
        of the total-variation
        super-resolution algorithm.

    num_primal_dual_loops : string
        Number of primal/dual (inner) loops used in the optimization of the
        total-variation super-resolution algorithm.

    num_bregman_loops : string
        Number of Bregman (outer) loops used in the optimization of the
        total-variation super-resolution algorithm.

    step_scale : string
        Step scale parameter used in the optimization of the total-variation
        super-resolution algorithm.

    gamma : string
        Gamma parameter used in the optimization of the total-variation
        super-resolution algorithm.

    sr_id : string
        ID of the reconstruction useful to distinguish when multiple
        reconstructions with different order of stacks are run on
        the same subject

    session : string
        Session ID if applicable (in the form ``ses-YY``)

    m_stacks : list(int)
        List of stack to be used in the reconstruction.
        The specified order is kept if `skip_stacks_ordering` is True.

    m_masks_derivatives_dir : string
        directory basename in BIDS directory derivatives where to search
        for masks (optional)

    m_skip_svr : bool
        Weither the Slice-to-Volume Registration should be skipped in the
        image reconstruction. (default is False)

    m_do_refine_hr_mask : bool
        Weither a refinement of the HR mask should be performed.
        (default is False)

    m_do_nlm_denoising : bool
        Weither the NLM denoising preprocessing should be performed prior to
        motion estimation. (default is False)

    m_skip_stacks_ordering : bool (optional)
        Weither the automatic stacks ordering should be skipped.
        (default is False)

    Examples
    --------
    >>> from pymialsrtk.pipelines.anatomical.srr import SRReconPipeline
    >>> # Create a new instance
    >>> pipeline = SRReconPipeline(bids_dir='/path/to/bids_dir',
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

    def __init__(
        self, bids_dir, output_dir, subject, p_ga=None, p_stacks=None, sr_id=1,
        session=None, paramTV=None, p_masks_derivatives_dir=None,
        p_masks_desc=None, p_dict_custom_interfaces=None,
        openmp_number_of_cores=None, nipype_number_of_cores=None
    ):
        """Constructor of Preprocessing class instance."""

        super().__init__(bids_dir, output_dir, subject, p_ga, p_stacks, sr_id,
                         session, paramTV, p_masks_derivatives_dir,
                         p_masks_desc, p_dict_custom_interfaces,
                         openmp_number_of_cores, nipype_number_of_cores)

    def create_workflow(self):
        """Create the Niype workflow of the super-resolution pipeline.

        It is composed of a succession of Nodes and their corresponding parameters,
        where the output of node i goes to the input of node i+1.

        """


        self.wf = pe.Workflow(name=self.pipeline_name,
                              base_dir=self.wf_base_dir
                              )

        config.update_config(
            {
                'logging': {
                      'log_directory': os.path.join(self.wf_base_dir),
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
            self.bids_dir,
            self.subject,
            self.session,
            self.use_manual_masks,
            self.m_masks_desc,
            self.m_masks_derivatives_dir,
            self.m_skip_stacks_ordering,
            self.m_stacks
        )

        preprocessing_stage = preproc_stage.create_preproc_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising)

        reconstruction_stage = recon_stage.create_recon_stage(
            p_paramTV=self.paramTV,
            p_use_manual_masks=self.use_manual_masks,
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_do_refine_hr_mask=self.m_do_refine_hr_mask,
            p_skip_svr=self.m_skip_svr,
            p_sub_ses=self.sub_ses)

        postprocessing_stage = postproc_stage.create_postproc_stage(
            p_ga=self.m_ga,
            p_do_anat_orientation=self.m_do_anat_orientation,
            name='postprocessing_stage')

        output_mgmt_stage = srr_output_stage.create_srr_output_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_skip_stacks_ordering=self.m_skip_stacks_ordering,
            name='output_mgmt_stage')

        output_mgmt_stage.inputs.inputnode.sub_ses = self.sub_ses
        output_mgmt_stage.inputs.inputnode.sr_id = self.sr_id
        output_mgmt_stage.inputs.inputnode.use_manual_masks = \
            self.use_manual_masks
        output_mgmt_stage.inputs.inputnode.final_res_dir = self.final_res_dir


        # Build workflow : connections of the nodes
        # Nodes ready : Linking now
        self.wf.connect(input_stage, "outputnode.t2ws_filtered",
                        preprocessing_stage, "inputnode.input_images")

        self.wf.connect(input_stage, "outputnode.masks_filtered",
                        preprocessing_stage, "inputnode.input_masks")

        if self.m_do_nlm_denoising:
            self.wf.connect(preprocessing_stage,
                            ("outputnode.output_images_nlm",
                             utils.sort_ascending),
                            reconstruction_stage, "inputnode.input_images_nlm")

        self.wf.connect(preprocessing_stage,
                        ("outputnode.output_images", utils.sort_ascending),
                        reconstruction_stage, "inputnode.input_images")

        self.wf.connect(preprocessing_stage,
                        ("outputnode.output_masks", utils.sort_ascending),
                        reconstruction_stage, "inputnode.input_masks")

        self.wf.connect(input_stage, "outputnode.stacks_order",
                        reconstruction_stage, "inputnode.stacks_order")

        self.wf.connect(reconstruction_stage, "outputnode.output_hr_mask",
                        postprocessing_stage, "inputnode.input_mask")

        self.wf.connect(reconstruction_stage, "outputnode.output_sr",
                        postprocessing_stage, "inputnode.input_image")

        self.wf.connect(reconstruction_stage, "outputnode.output_sdi",
                        postprocessing_stage, "inputnode.input_sdi")

        self.wf.connect(input_stage, "outputnode.stacks_order",
                        output_mgmt_stage, "inputnode.stacks_order")

        self.wf.connect(preprocessing_stage, "outputnode.output_masks",
                        output_mgmt_stage, "inputnode.input_masks")
        self.wf.connect(preprocessing_stage, "outputnode.output_images",
                        output_mgmt_stage, "inputnode.input_images")
        self.wf.connect(reconstruction_stage, "outputnode.output_transforms",
                        output_mgmt_stage, "inputnode.input_transforms")

        self.wf.connect(reconstruction_stage, "outputnode.output_sdi",
                        output_mgmt_stage, "inputnode.input_sdi")
        self.wf.connect(postprocessing_stage, "outputnode.output_image",
                        output_mgmt_stage, "inputnode.input_sr")
        self.wf.connect(reconstruction_stage, "outputnode.output_json_path",
                        output_mgmt_stage, "inputnode.input_json_path")
        self.wf.connect(reconstruction_stage, "outputnode.output_sr_png",
                        output_mgmt_stage, "inputnode.input_sr_png")
        self.wf.connect(postprocessing_stage, "outputnode.output_mask",
                        output_mgmt_stage, "inputnode.input_hr_mask")

        if self.m_do_nlm_denoising:
            self.wf.connect(preprocessing_stage,
                            "outputnode.output_images_nlm",
                            output_mgmt_stage, "inputnode.input_images_nlm")

        if not self.m_skip_stacks_ordering:
            self.wf.connect(input_stage, "outputnode.report_image",
                            output_mgmt_stage, "inputnode.report_image")
            self.wf.connect(input_stage, "outputnode.motion_tsv",
                            output_mgmt_stage, "inputnode.motion_tsv")