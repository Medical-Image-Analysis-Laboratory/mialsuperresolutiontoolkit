# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital
# Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Abstract base class for the anatomical pipeline."""

import abc
import os
import shutil
from datetime import datetime
from nipype.info import __version__ as __nipype_version__
from nipype import logging as nipype_logging

# Import the implemented interface from pymialsrtk
from pymialsrtk.bids.utils import write_bids_derivative_description

# Get pymialsrtk version
from pymialsrtk.info import __version__


class AbstractAnatomicalPipeline:
    """Class used to represent the workflow of the
    anatomical pipeline.

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

    m_do_nlm_denoising : bool
        Whether the NLM denoising preprocessing should be performed prior to
        motion estimation. (default is False)

    m_skip_stacks_ordering : bool (optional)
        Whether the automatic stacks ordering should be skipped.
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
                                      paramTV={deltat_TV = "0.001",
                                               lambda_TV = "0.75",
                                               num_primal_dual_loops = "20"},
                                      masks_derivatives_dir="/custom/mask_dir",
                                      masks_desc=None,
                                      p_dict_custom_interfaces=None)
    >>> # Create the super resolution Nipype workflow
    >>> pipeline.create_workflow()
    >>> # Execute the workflow
    >>> res = pipeline.run(number_of_cores=1) # doctest: +SKIP

    """

    m_pipeline_name = None
    m_run_start_time = None
    m_run_end_time = None
    m_run_elapsed_time = None

    m_bids_dir = None
    m_output_dir = None
    m_subject = None
    m_wf = None
    m_sr_id = None
    m_session = None
    m_stacks = None

    m_masks_derivatives_dir = None
    m_use_manual_masks = False
    m_masks_desc = None

    m_verbose = None
    m_openmp_number_of_cores = None
    m_nipype_number_of_cores = None

    m_sub_ses = None
    m_sub_path = None
    m_wf_base_dir = None
    m_final_res_dir = None

    def __init__(
        self,
        p_bids_dir,
        p_output_dir,
        p_subject,
        p_ga=None,
        p_stacks=None,
        p_sr_id=1,
        p_session=None,
        p_masks_derivatives_dir=None,
        p_masks_desc=None,
        p_dict_custom_interfaces=None,
        p_verbose=None,
        p_openmp_number_of_cores=None,
        p_nipype_number_of_cores=None,
        p_run_type=None,
    ):
        """Constructor of AnatomicalPipeline class instance."""

        # BIDS processing parameters
        self.m_bids_dir = p_bids_dir
        self.m_output_dir = p_output_dir
        self.m_subject = p_subject
        self.m_ga = p_ga
        self.m_sr_id = p_sr_id
        self.m_session = p_session
        self.m_stacks = p_stacks
        self.m_run_type = p_run_type
        self.m_verbose = p_verbose
        self.m_openmp_number_of_cores = p_openmp_number_of_cores
        self.m_nipype_number_of_cores = p_nipype_number_of_cores

        # Use manual/custom brain masks
        # If masks directory is not specified use the
        # automated brain extraction method.
        self.m_masks_derivatives_dir = p_masks_derivatives_dir
        self.m_use_manual_masks = (
            True if self.m_masks_derivatives_dir is not None else False
        )
        self.m_masks_desc = p_masks_desc if self.m_use_manual_masks else None

        self.m_sub_ses = self.m_subject
        self.m_sub_path = self.m_subject

        if self.m_session is not None:
            self.m_sub_ses = "".join([self.m_sub_ses, "_", self.m_session])
            self.m_sub_path = os.path.join(self.m_subject, self.m_session)

        self.m_wf_base_dir = os.path.join(
            self.m_output_dir,
            "-".join(["nipype", __nipype_version__]),
            self.m_sub_path,
            f"{self.m_run_type}-{self.m_sr_id}",
        )

        self.m_final_res_dir = os.path.join(
            self.m_output_dir,
            "-".join(["pymialsrtk", __version__]),
            self.m_sub_path,
        )

        if not os.path.exists(self.m_wf_base_dir):
            os.makedirs(self.m_wf_base_dir)
        print("Process directory: {}".format(self.m_wf_base_dir))

        # Initialization (Not sure we can control the name of nipype log)
        if os.path.isfile(os.path.join(self.m_wf_base_dir, "pypeline.log")):
            os.unlink(os.path.join(self.m_wf_base_dir, "pypeline.log"))

    @abc.abstractmethod
    def create_workflow(self):
        """Create the Niype workflow of the super-resolution pipeline.

        It is composed of a succession of Nodes and their corresponding
        parameters, where the output of node i goes to the input of node i+1.

        The more specific definition given in each node implementing
        the method.
        """

    def run(self, memory=None, logger=None):
        """Execute the workflow of the super-resolution
        reconstruction pipeline.

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
        if logger:
            iflogger = logger
        else:
            iflogger = nipype_logging.getLogger("nipype.interface")
        iflogger.info("**** Workflow graph creation ****")
        self.m_wf.write_graph(
            dotfilename="graph.dot",
            graph2use="colored",
            format="png",
            simple_form=True,
        )

        # Copy and rename the generated "graph.png" image
        src = os.path.join(self.m_wf.base_dir, self.m_wf.name, "graph.png")

        # String formatting for saving
        subject_str = f"{self.m_subject}"
        dst_base = os.path.join(
            self.m_output_dir,
            "-".join(["pymialsrtk", __version__]),
            self.m_subject,
        )

        if self.m_session is not None:
            subject_str += f"_{self.m_session}"
            dst_base = os.path.join(dst_base, self.m_session)

        dst = os.path.join(
            dst_base,
            "figures",
            f"{subject_str}_{self.m_run_type}-SR_id-{self.m_sr_id}_"
            + "desc-processing_graph.png",
        )

        # Create the figures/ and parent directories if they do not exist
        figures_dir = os.path.dirname(dst)
        os.makedirs(figures_dir, exist_ok=True)
        # Make the copy
        iflogger.info(f"\t > Copy {src} to {dst}...")
        shutil.copy(src=src, dst=dst)

        # Create dictionary of arguments passed to plugin_args
        args_dict = {
            "raise_insufficient": False,
            "n_procs": self.m_nipype_number_of_cores,
        }

        if (memory is not None) and (memory > 0):
            args_dict["memory_gb"] = memory

        iflogger.info("**** Processing ****")
        # datetime object containing current start date and time
        start = datetime.now()
        self.m_run_start_time = start.strftime("%B %d, %Y / %H:%M:%S")
        print(f" Start date / time : {self.m_run_start_time}")

        # Execute the workflow
        if self.m_nipype_number_of_cores > 1:
            res = self.m_wf.run(plugin="MultiProc", plugin_args=args_dict)
        else:
            res = self.m_wf.run()

        # Copy and rename the workflow execution log
        src = os.path.join(self.m_wf.base_dir, "pypeline.log")
        dst = os.path.join(
            dst_base,
            "logs",
            f"{subject_str}_{self.m_run_type}-SR_id-{self.m_sr_id}_log.txt",
        )
        # Create the logs/ and parent directories if they do not exist
        logs_dir = os.path.dirname(dst)
        os.makedirs(logs_dir, exist_ok=True)
        # Make the copy
        iflogger.info(f"\t > Copy {src} to {dst}...")
        shutil.copy(src=src, dst=dst)

        # datetime object containing current end date and time
        end = datetime.now()
        self.m_run_end_time = end.strftime("%B %d, %Y / %H:%M:%S")
        print(f" End date / time : {self.m_run_end_time}")

        # Compute elapsed running time in minutes and seconds
        duration = end - start
        (minutes, seconds) = divmod(duration.total_seconds(), 60)
        self.m_run_elapsed_time = f"{int(minutes)} min. and {int(seconds)} s."
        print(f" Elapsed time: {self.m_run_end_time}")

        iflogger.info("**** Write dataset derivatives description ****")
        for toolbox in ["pymialsrtk", "nipype"]:
            write_bids_derivative_description(
                bids_dir=self.m_bids_dir,
                deriv_dir=self.m_output_dir,
                pipeline_name=toolbox,
            )

        return res
