# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the super-resolution reconstruction pipeline."""

import os
import sys
import platform
import json
import pkg_resources
from jinja2 import Environment, FileSystemLoader
from jinja2 import __version__ as __jinja2_version__
import nibabel as nib
import pymialsrtk.interfaces.utils as utils
from nipype.info import __version__ as __nipype_version__
from nipype import config
from nipype import logging as nipype_logging
from nipype.pipeline import engine as pe

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.workflows.preproc_stage as preproc_stage
import pymialsrtk.workflows.postproc_stage as postproc_stage
import pymialsrtk.workflows.srr_assessment_stage as srr_assment_stage
import pymialsrtk.workflows.recon_stage as recon_stage
import pymialsrtk.workflows.output_stage as output_stage
import pymialsrtk.workflows.input_stage as input_stage

from .abstract import AbstractAnatomicalPipeline

# Get pymialsrtk version
from pymialsrtk.info import __version__


class SRReconPipeline(AbstractAnatomicalPipeline):
    """Class used to represent the workflow of the Super-Resolution
    reconstruction pipeline.

    Attributes
    -----------
    m_bids_dir : string
        BIDS root directory (required)

    m_output_dir : string
        Output derivatives directory (required)

    m_subject : string
        Subject ID (in the form ``sub-XX``)

    m_wf : nipype.pipeline.Workflow
        Nipype workflow of the reconstruction pipeline

    m_paramTV: :obj:`dict`
        Dictionary of parameters for the super-resolution
        reconstruction. Contains:
        - deltatTV : string
            Super-resolution optimization time-step
        - lambdaTV : float
            Regularization weight (default is 0.75)
        - num_iterations : string
            Number of iterations in the primal/dual loops used in the
            optimization of the total-variation super-resolution algorithm.
        - num_primal_dual_loops : string
            Number of primal/dual (inner) loops used in the optimization of the
            total-variation super-resolution algorithm.
        - num_bregman_loops : string
            Number of Bregman (outer) loops used in the optimization of the
            total-variation super-resolution algorithm.
        - step_scale : string
            Step scale parameter used in the optimization of the total-
            variation super-resolution algorithm.
        - gamma : string
            Gamma parameter used in the optimization of the total-variation
            super-resolution algorithm.

    m_sr_id : string
        ID of the reconstruction useful to distinguish when multiple
        reconstructions with different order of stacks are run on
        the same subject

    m_session : string
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
    m_pipeline_name = "srr_pipeline"
    m_paramTV = None
    m_labels_derivatives_dir = None
    m_paramTV = None

    # Custom interfaces options
    m_do_nlm_denoising = None
    m_skip_stacks_ordering = None
    m_skip_svr = None
    m_do_refine_hr_mask = None
    m_do_anat_orientation = None
    m_do_multi_parameters = None
    m_do_srr_assessment = None

    def __init__(
        self,
        p_bids_dir,
        p_output_dir,
        p_subject,
        p_ga=None,
        p_stacks=None,
        p_sr_id=1,
        p_session=None,
        p_paramTV=None,
        p_masks_derivatives_dir=None,
        p_labels_derivatives_dir=None,
        p_masks_desc=None,
        p_dict_custom_interfaces=None,
        p_verbose=None,
        p_openmp_number_of_cores=None,
        p_nipype_number_of_cores=None
    ):
        """Constructor of SRReconPipeline class instance."""

        super().__init__(p_bids_dir, p_output_dir, p_subject, p_ga, p_stacks,
                         p_sr_id, p_session, p_masks_derivatives_dir,
                         p_masks_desc, p_dict_custom_interfaces, p_verbose,
                         p_openmp_number_of_cores, p_nipype_number_of_cores,
                         "rec"
                         )
        # (default) sr tv parameters
        if p_paramTV is None:
            p_paramTV = dict()
        self.m_paramTV = p_paramTV

        self.m_labels_derivatives_dir = p_labels_derivatives_dir

        # Custom interfaces and default values.
        if p_dict_custom_interfaces is not None:
            self.m_do_nlm_denoising = \
                p_dict_custom_interfaces['do_nlm_denoising'] \
                if 'do_nlm_denoising' in p_dict_custom_interfaces.keys() \
                else False

            self.m_skip_stacks_ordering =\
                p_dict_custom_interfaces['skip_stacks_ordering']\
                if ((self.m_stacks is not None) and
                    ('skip_stacks_ordering' in
                     p_dict_custom_interfaces.keys())) \
                else False

            self.m_skip_svr = \
                p_dict_custom_interfaces['skip_svr'] \
                if 'skip_svr' in p_dict_custom_interfaces.keys() \
                else False

            self.m_do_refine_hr_mask = \
                p_dict_custom_interfaces['do_refine_hr_mask'] \
                if 'do_refine_hr_mask' in p_dict_custom_interfaces.keys() \
                else False

            self.m_do_anat_orientation = \
                p_dict_custom_interfaces['do_anat_orientation'] \
                if 'do_anat_orientation' in p_dict_custom_interfaces.keys() \
                else False

            self.m_do_reconstruct_labels = \
                p_dict_custom_interfaces['do_reconstruct_labels'] \
                if 'do_reconstruct_labels' in p_dict_custom_interfaces.keys() \
                else False

            self.m_do_multi_parameters = \
                p_dict_custom_interfaces['do_multi_parameters'] \
                if 'do_multi_parameters' in p_dict_custom_interfaces.keys() \
                else False

            self.m_do_srr_assessment = \
                p_dict_custom_interfaces['do_srr_assessment'] \
                if 'do_srr_assessment' in p_dict_custom_interfaces.keys() \
                else False

        else:

            self.m_do_nlm_denoising = False
            self.m_skip_stacks_ordering = False
            self.m_skip_svr = False
            self.m_do_refine_hr_mask = False
            self.m_do_reconstruct_labels = False
            self.m_do_anat_orientation = False
            self.m_do_multi_parameters = False
            self.m_do_srr_assessment = False

        if self.m_do_anat_orientation:
            if not os.path.isdir('/sta'):
                raise RuntimeError(
                    'A template directory must be specified to '
                    'perform alignement.')
            if self.m_ga is None:
                raise RuntimeError(
                    'A gestational age must be specified to '
                    'perform alignement.'
                      )

        if self.m_do_reconstruct_labels:
            if not self.m_labels_derivatives_dir:
                raise OSError(
                    'A derivatives directory of LR labelmaps must '
                    'be specified to perform labelmap reconstruction.'
                    )
            elif not os.path.isdir(
                os.path.join(self.m_bids_dir,
                             'derivatives',
                             self.m_labels_derivatives_dir
                             )):
                raise OSError(
                    'An existing derivatives directory of LR labelmaps must'
                    'be specified to perform labelmap reconstruction.'
                    )

        if self.m_do_multi_parameters:
            # if any of the TV parameters is a list of more than one item,
            # we are in a multi_parameters running mode
            num_parameters_multi = [value
                                    for value in list(self.m_paramTV.values())
                                    if (isinstance(value, list)
                                        and len(value) > 1)]
            if not num_parameters_multi:
                raise RuntimeError(
                    'With do_multi_parameters interface, '
                    'at least one entry of \'paramsTV\' should '
                    'be defined as a list of more than one item.'
                )
        else:
            num_parameters_multi = [value
                                    for value in list(self.m_paramTV.values())
                                    if (isinstance(value, list)
                                        and len(value) > 1)]
            if num_parameters_multi:
                raise RuntimeError(
                    'With do_multi_parameters=False, '
                    'no entry of \'paramsTV\' should '
                    'be defined as a list of more than one item.'
                )

        if self.m_do_anat_orientation and self.m_do_srr_assessment:
            raise RuntimeError(
                'At the moment, srr pipeline with custom interfaces '
                'do_anat_orientation and do_srr_assessment '
                'is not available.'
            )

        if not self.m_use_manual_masks and self.m_do_srr_assessment:
            raise RuntimeError(
                'do_srr_assessment interface requires'
                'to provide low-resolution binary masks.'
            )

    def create_workflow(self):
        """Create the Niype workflow of the super-resolution pipeline.

        It is composed of a succession of Nodes and their corresponding parameters,
        where the output of node i goes to the input of node i+1.

        """

        self.m_wf = pe.Workflow(name=self.m_pipeline_name,
                                base_dir=self.m_wf_base_dir
                                )

        self.m_wf.config['logging'] = {
            'log_directory': os.path.join(self.m_wf_base_dir),
            'log_to_file': True
        }
        self.m_wf.config['execution'] = {
            'remove_unnecessary_outputs': False,
            'stop_on_first_crash': True,
            'stop_on_first_rerun': True,
            'crashfile_format': "txt",
            'use_relative_paths': True,
            'write_provenance': False
        }

        config.update_config(self.m_wf.config)

        # Update nypipe logging with config
        nipype_logging.update_logging(config)
        # config.enable_provenance()

        input_mgmt_stage = input_stage.create_input_stage(
                p_bids_dir=self.m_bids_dir,
                p_subject=self.m_subject,
                p_session=self.m_session,
                p_use_manual_masks=self.m_use_manual_masks,
                p_masks_desc=self.m_masks_desc,
                p_masks_derivatives_dir=self.m_masks_derivatives_dir,
                p_labels_derivatives_dir=self.m_labels_derivatives_dir,
                p_skip_stacks_ordering=self.m_skip_stacks_ordering,
                p_do_reconstruct_labels=self.m_do_reconstruct_labels,
                p_stacks=self.m_stacks,
                p_do_srr_assessment=self.m_do_srr_assessment,
                p_verbose=self.m_verbose,
                name='input_mgmt_stage'
                )

        preprocessing_stage = preproc_stage.create_preproc_stage(
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_do_reconstruct_labels=self.m_do_reconstruct_labels,
            p_verbose=self.m_verbose
            )

        reconstruction_stage, srtv_node_name = recon_stage.create_recon_stage(
            p_paramTV=self.m_paramTV,
            p_use_manual_masks=self.m_use_manual_masks,
            p_do_multi_parameters=self.m_do_multi_parameters,
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_do_reconstruct_labels=self.m_do_reconstruct_labels,
            p_do_refine_hr_mask=self.m_do_refine_hr_mask,
            p_skip_svr=self.m_skip_svr,
            p_sub_ses=self.m_sub_ses,
            p_verbose=self.m_verbose)


        postprocessing_stage = postproc_stage.create_postproc_stage(
            p_ga=self.m_ga,
            p_do_anat_orientation=self.m_do_anat_orientation,
            p_do_reconstruct_labels=self.m_do_reconstruct_labels,
            p_verbose=self.m_verbose,
            name='postprocessing_stage')

        if self.m_do_srr_assessment:
            srr_assessment_stage = \
                srr_assment_stage.create_srr_assessment_stage(
                    p_do_multi_parameters=self.m_do_multi_parameters,
                    p_input_srtv_node=srtv_node_name,
                    p_verbose=self.m_verbose,
                    p_openmp_number_of_cores=self.m_openmp_number_of_cores,
                    name='srr_assessment_stage'
                )

        output_mgmt_stage = output_stage.create_srr_output_stage(
            p_sub_ses=self.m_sub_ses,
            p_sr_id=self.m_sr_id,
            p_run_type=self.m_run_type,
            p_use_manual_masks=self.m_use_manual_masks,
            p_do_nlm_denoising=self.m_do_nlm_denoising,
            p_do_reconstruct_labels=self.m_do_reconstruct_labels,
            p_skip_stacks_ordering=self.m_skip_stacks_ordering,
            p_do_srr_assessment=self.m_do_srr_assessment,
            name='output_mgmt_stage'
        )
        output_mgmt_stage.inputs.inputnode.final_res_dir = self.m_final_res_dir

        # Build workflow : connections of the nodes
        # Nodes ready : Linking now
        self.m_wf.connect(input_mgmt_stage, "outputnode.t2ws_filtered",
                          preprocessing_stage, "inputnode.input_images")

        self.m_wf.connect(input_mgmt_stage, "outputnode.masks_filtered",
                          preprocessing_stage, "inputnode.input_masks")

        if self.m_do_nlm_denoising:
            self.m_wf.connect(preprocessing_stage,
                              "outputnode.output_images_nlm",
                              reconstruction_stage,
                              "inputnode.input_images_nlm")

        self.m_wf.connect(preprocessing_stage, "outputnode.output_images",
                          reconstruction_stage, "inputnode.input_images")

        self.m_wf.connect(preprocessing_stage, "outputnode.output_masks",
                          reconstruction_stage, "inputnode.input_masks")

        self.m_wf.connect(input_mgmt_stage, "outputnode.stacks_order",
                          reconstruction_stage, "inputnode.stacks_order")

        self.m_wf.connect(reconstruction_stage, "outputnode.output_hr_mask",
                          postprocessing_stage, "inputnode.input_mask")

        self.m_wf.connect(reconstruction_stage, "outputnode.output_sr",
                          postprocessing_stage, "inputnode.input_image")

        if self.m_do_reconstruct_labels:
            self.m_wf.connect(input_mgmt_stage, "outputnode.labels_filtered",
                              preprocessing_stage, "inputnode.input_labels")

            self.m_wf.connect(preprocessing_stage, "outputnode.output_labels",
                              reconstruction_stage, "inputnode.input_labels")

            self.m_wf.connect(reconstruction_stage,
                              "outputnode.output_labelmap",
                              postprocessing_stage,
                              "inputnode.input_labelmap")

            self.m_wf.connect(postprocessing_stage,
                              "outputnode.output_labelmap",
                              output_mgmt_stage,
                              "inputnode.input_labelmap")

        if self.m_do_anat_orientation:
            self.m_wf.connect(reconstruction_stage, "outputnode.output_sdi",
                              postprocessing_stage, "inputnode.input_sdi")

        if self.m_do_srr_assessment:
            self.m_wf.connect(reconstruction_stage,
                              "outputnode.output_TV_parameters",
                              srr_assessment_stage,
                              "inputnode.input_TV_parameters")

            if self.m_do_anat_orientation:
                self.m_wf.connect(postprocessing_stage,
                                  "outputnode.output_sdi",
                                  srr_assessment_stage,
                                  "inputnode.input_sdi_image")
            else:
                self.m_wf.connect(reconstruction_stage,
                                  "outputnode.output_sdi",
                                  srr_assessment_stage,
                                  "inputnode.input_sdi_image")

            self.m_wf.connect(postprocessing_stage,
                              "outputnode.output_image",
                              srr_assessment_stage,
                              "inputnode.input_sr_image")

            self.m_wf.connect(input_mgmt_stage,
                              "outputnode.hr_reference_image",
                              srr_assessment_stage,
                              "inputnode.input_ref_image")

            self.m_wf.connect(input_mgmt_stage,
                              "outputnode.hr_reference_mask",
                              srr_assessment_stage,
                              "inputnode.input_ref_mask")

            self.m_wf.connect(input_mgmt_stage,
                              "outputnode.hr_reference_labels",
                              srr_assessment_stage,
                              "inputnode.input_ref_labelmap")

            self.m_wf.connect(srr_assessment_stage,
                              "outputnode.output_metrics",
                              output_mgmt_stage,
                              "inputnode.input_metrics")

            self.m_wf.connect(srr_assessment_stage,
                              "outputnode.output_metrics_labels",
                              output_mgmt_stage,
                              "inputnode.input_metrics_labels")

        self.m_wf.connect(input_mgmt_stage, "outputnode.stacks_order",
                          output_mgmt_stage, "inputnode.stacks_order")

        self.m_wf.connect(preprocessing_stage, "outputnode.output_masks",
                          output_mgmt_stage, "inputnode.input_masks")
        self.m_wf.connect(preprocessing_stage, "outputnode.output_images",
                          output_mgmt_stage, "inputnode.input_images")
        self.m_wf.connect(reconstruction_stage, "outputnode.output_transforms",
                          output_mgmt_stage, "inputnode.input_transforms")

        self.m_wf.connect(reconstruction_stage, "outputnode.output_sdi",
                          output_mgmt_stage, "inputnode.input_sdi")
        self.m_wf.connect(postprocessing_stage, "outputnode.output_image",
                          output_mgmt_stage, "inputnode.input_sr")
        self.m_wf.connect(reconstruction_stage, "outputnode.output_json_path",
                          output_mgmt_stage, "inputnode.input_json_path")
        self.m_wf.connect(reconstruction_stage, "outputnode.output_sr_png",
                          output_mgmt_stage, "inputnode.input_sr_png")
        self.m_wf.connect(postprocessing_stage, "outputnode.output_mask",
                          output_mgmt_stage, "inputnode.input_hr_mask")

        if self.m_do_nlm_denoising:
            self.m_wf.connect(preprocessing_stage,
                              "outputnode.output_images_nlm",
                              output_mgmt_stage, "inputnode.input_images_nlm")

        if not self.m_skip_stacks_ordering:
            self.m_wf.connect(input_mgmt_stage, "outputnode.report_image",
                              output_mgmt_stage, "inputnode.report_image")
            self.m_wf.connect(input_mgmt_stage, "outputnode.motion_tsv",
                              output_mgmt_stage, "inputnode.motion_tsv")

    def run(self, memory=None):
        iflogger = nipype_logging.getLogger('nipype.interface')
        res = super().run(memory, iflogger)

        if not self.m_do_multi_parameters:
            iflogger.info("**** Super-resolution HTML report creation ****")
            self.create_subject_report()

        return res

    def create_subject_report(self):
        """Create the HTML report"""
        # Set main subject derivatives directory
        sub_ses = self.m_subject
        sub_path = self.m_subject
        if self.m_session is not None:
            sub_ses += f'_{self.m_session}'
            sub_path = os.path.join(self.m_subject, self.m_session)

        final_res_dir = os.path.join(self.m_output_dir,
                                     '-'.join(["pymialsrtk", __version__]),
                                     sub_path)

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
            f'{sub_ses}_{self.m_run_type}-SR_id-{self.m_sr_id}_T2w.nii.gz'
        )
        img = nib.load(sr_nii_image)
        sx, sy, sz = img.header.get_zooms()

        sr_json_metadata = os.path.join(
            final_res_dir, 'anat',
            f'{sub_ses}_{self.m_run_type}-SR_id-{self.m_sr_id}_T2w.json'
        )
        with open(sr_json_metadata) as f:
            sr_json_metadata = json.load(f)

        workflow_image = os.path.join(
            '..', 'figures',
            f'{sub_ses}_{self.m_run_type}-SR_id-'
            f'{self.m_sr_id}_desc-processing_graph.png'
        )

        sr_png_image = os.path.join(
            '..', 'figures',
            f'{sub_ses}_{self.m_run_type}-SR_id-{self.m_sr_id}_T2w.png'
        )

        motion_report_image = os.path.join(
            '..', 'figures',
            f'{sub_ses}_{self.m_run_type}-SR_id-'
            f'{self.m_sr_id}_desc-motion_stats.png'
        )

        log_file = os.path.join(
            '..', 'logs',
            f'{sub_ses}_{self.m_run_type}-SR_id-{self.m_sr_id}_log.txt'
        )

        # Create the text for {{subject}} and {{session}} fields in template
        report_subject_text = f'{self.m_subject.split("-")[-1]}'
        if self.m_session is not None:
            report_session_text = f'{self.m_session.split("-")[-1]}'
        else:
            report_session_text = None

        # Generate the report
        report_html_content = template.render(
            subject=report_subject_text,
            session=report_session_text,
            processing_datetime=self.m_run_start_time,
            run_time=self.m_run_elapsed_time,
            log=log_file,
            sr_id=self.m_sr_id,
            stacks=self.m_stacks,
            svr="on" if not self.m_skip_svr else "off",
            nlm_denoising="on" if self.m_do_nlm_denoising else "off",
            stacks_ordering="on" if not self.m_skip_stacks_ordering else "off",
            do_refine_hr_mask="on" if self.m_do_refine_hr_mask else "off",
            use_auto_masks="on" if self.m_masks_derivatives_dir is None
                else "off",
            custom_masks_dir=self.m_masks_derivatives_dir
                if self.m_masks_derivatives_dir is not None else None,
            sr_resolution=f"{sx} x {sy} x {sz} mm<sup>3</sup>",
            sr_json_metadata=sr_json_metadata,
            workflow_graph=workflow_image,
            sr_png_image=sr_png_image,
            motion_report_image=motion_report_image,
            version=__version__,
            os=f'{platform.system()} {platform.release()}',
            python=f'{sys.version}',
            openmp_threads=self.m_openmp_number_of_cores,
            nipype_threads=self.m_nipype_number_of_cores,
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
