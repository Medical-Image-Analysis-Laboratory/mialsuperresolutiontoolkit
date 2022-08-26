# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Abstract base class for the anatomical pipeline."""

import abc
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
from nipype import logging as nipype_logging

# Import the implemented interface from pymialsrtk
from pymialsrtk.bids.utils import write_bids_derivative_description

# Get pymialsrtk version
from pymialsrtk.info import __version__


class AnatomicalPipeline:
    """Class used to represent the workflow of the 
    anatomical pipeline.

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
    m_do_anat_orientation = None

    m_masks_derivatives_dir = None
    use_manual_masks = False
    m_masks_desc = None

    openmp_number_of_cores = None
    nipype_number_of_cores = None

    sub_ses = None
    sub_path = None
    wf_base_dir = None
    final_res_dir = None

    def __init__(
        self, bids_dir, output_dir, subject, p_ga=None, p_stacks=None, sr_id=1,
        session=None, paramTV=None, p_masks_derivatives_dir=None,
        p_masks_desc=None, p_dict_custom_interfaces=None,
        openmp_number_of_cores=None, nipype_number_of_cores=None
    ):
        """Constructor of AnatomicalPipeline class instance."""

        # BIDS processing parameters
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.subject = subject
        self.m_ga = p_ga
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
            self.m_skip_svr = p_dict_custom_interfaces['skip_svr'] \
                if 'skip_svr' in p_dict_custom_interfaces.keys() \
                else False
            self.m_do_refine_hr_mask = \
                p_dict_custom_interfaces['do_refine_hr_mask'] \
                if 'do_refine_hr_mask' in p_dict_custom_interfaces.keys() \
                else False
            self.m_do_nlm_denoising = p_dict_custom_interfaces['do_nlm_denoising']\
                if 'do_nlm_denoising' in p_dict_custom_interfaces.keys() \
                else False

            self.m_skip_stacks_ordering =\
                p_dict_custom_interfaces['skip_stacks_ordering']\
                    if ((self.m_stacks is not None) and
                        ('skip_stacks_ordering' in
                         p_dict_custom_interfaces.keys())) \
                    else False

            self.m_do_anat_orientation = \
                p_dict_custom_interfaces['do_anat_orientation'] \
                if 'do_anat_orientation' in p_dict_custom_interfaces.keys() \
                else False

        else:
            self.m_skip_svr = False
            self.m_do_refine_hr_mask = False
            self.m_do_nlm_denoising = False
            self.m_skip_stacks_ordering = False
            self.m_do_anat_orientation = False

        if self.m_do_anat_orientation:
            if not os.path.isdir('/sta'):
                print('A template directory must '
                      'be specified to perform alignement.')
                self.m_do_anat_orientation = False
            if self.m_ga is None:
                print('A gestational age must '
                      'be specified to perform alignement.')
                self.m_do_anat_orientation = False

        self.sub_ses = self.subject
        self.sub_path = self.subject
        if self.session is not None:
            self.sub_ses = ''.join([self.sub_ses, '_', self.session])
            self.sub_path = os.path.join(self.subject, self.session)

        self.wf_base_dir = os.path.join(
            self.output_dir,
            '-'.join(["nipype", __nipype_version__]),
            self.sub_path,
            "rec-{}".format(self.sr_id)
            )

        self.final_res_dir = os.path.join(
            self.output_dir,
            '-'.join(["pymialsrtk", __version__]),
            self.sub_path
            )

        if not os.path.exists(self.wf_base_dir):
            os.makedirs(self.wf_base_dir)
        print("Process directory: {}".format(self.wf_base_dir))

        # Initialization (Not sure we can control the name of nipype log)
        if os.path.isfile(os.path.join(self.wf_base_dir, "pypeline.log")):
            os.unlink(os.path.join(self.wf_base_dir, "pypeline.log"))
               
    @abc.abstractmethod
    def create_workflow(self):
        """Create the Niype workflow of the super-resolution pipeline.

        It is composed of a succession of Nodes and their corresponding parameters,
        where the output of node i goes to the input of node i+1.
        
        The more specific definition given in each node implementing the method.
        """
        pass

    def run(self, memory=None):
        """Execute the workflow of the super-resolution reconstruction pipeline.

        Nipype execution engine will take care of the management and
        execution of all processing steps involved in the super-resolution
        reconstruction pipeline. Note that the complete execution graph is
        saved as a PNG image to support transparency on the whole processing.

        Parameters
        ----------
        memory : int
            Maximal memory used by the workflow
        """

        # Use nipype.interface logger to print some information messages
        iflogger = nipype_logging.getLogger('nipype.interface')
        iflogger.info("**** Workflow graph creation ****")
        self.wf.write_graph(dotfilename='graph.dot', graph2use='colored',
                            format='png', simple_form=True)

        # Copy and rename the generated "graph.png" image
        src = os.path.join(self.wf.base_dir, self.wf.name, 'graph.png')

        # String formatting for saving
        subject_str = f"{self.subject}"
        dst_base = os.path.join(self.output_dir,
                                '-'.join(["pymialsrtk", __version__]),
                                self.subject)

        if self.session is not None:
            subject_str += f"_{self.session}"
            dst_base = os.path.join(dst_base, self.session)

        dst = os.path.join(dst_base, 'figures',
                           f'{subject_str}_rec-SR_id-{self.sr_id}_' +
                           'desc-processing_graph.png')

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
        dst = os.path.join(dst_base, 'logs',
                           f'{subject_str}_rec-SR_id-{self.sr_id}_log.txt')
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
        sub_ses = self.subject
        sub_path = self.subject
        if self.session is not None:
            sub_ses += f'_{self.session}'
            sub_path = os.path.join(self.subject, self.session)

        final_res_dir = os.path.join(self.output_dir,
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