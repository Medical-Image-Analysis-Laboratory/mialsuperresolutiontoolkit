# Copyright © 2016-2020 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""Module for the super-resolution reconstruction pipeline."""

import os

import pkg_resources

from nipype import config, logging
# from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.pipeline import Node, MapNode, Workflow
from nipype.interfaces.utility import IdentityInterface, Function


# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess
import pymialsrtk.interfaces.utils as utils

# Get pymialsrtk version
from pymialsrtk.info import __version__


class AnatomicalPipeline:
    """Class used to represent the workflow of the Super-Resolution reconstruction pipeline.

    Attributes
    -----------
    bids_dir <string>
        BIDS root directory (required)

    output_dir <string>
        Output derivatives directory (required)

    subject <string>
        Subject ID (in the form ``sub-XX``)

    wf <nipype.pipeline.Workflow>
        Nipype workflow of the reconstruction pipeline

    dictsink <nipype.interfaces.io.JSONFileSink>
        Nipype node used to generate a JSON file that store provenance metadata
        for the SR-reconstructed images

    deltatTV <string>
        Super-resolution optimization time-step

    lambdaTV <Float>
        Regularization weight (default is 0.75)

    primal_dual_loops <string>
        Number of primal/dual loops used in the optimization of the total-variation
        super-resolution algorithm.

    sr_id <string>
        ID of the reconstruction useful to distinguish when multiple reconstructions
        with different order of stacks are run on the same subject

    session <string>
        Session ID if applicable (in the form ``ses-YY``)

    m_stacks list<<int>>
        List of stack to be used in the reconstruction. The specified order is kept if `skip_stacks_ordering` is True.

    m_masks_derivatives_dir <string>
        directory basename in BIDS directory derivatives where to search for masks (optional)

    m_skip_svr <bool>
        Weither the Slice-to-Volume Registration should be skipped in the image reconstruction. (default is False)

    m_do_refine_hr_mask <bool>
        Weither a refinement of the HR mask should be performed. (default is False)

    m_skip_nlm_denoising <bool>
        Weither the NLM denoising preprocessing should be skipped. (default is False)

    m_skip_stacks_ordering <bool> (optional)
        Weither the automatic stacks ordering should be skipped. (default is False)


    Examples
    --------
    >>> from pymialsrtk.pipelines.anatomical.srr import AnatomicalPipeline
    >>> # Create a new instance
    >>> pipeline = AnatomicalPipeline('/path/to/bids_dir',
                                  '/path/to/output_dir',
                                  'sub-01',
                                  [1,3,2,0],
                                  01,
                                  None,
                                  paramTV={deltatTV = "0.001",
                                           lambdaTV = "0.75",
                                           primal_dual_loops = "20"},
                                  use_manual_masks=False)
    >>> # Create the super resolution Nipype workflow
    >>> pipeline.create_workflow()
    >>> # Execute the workflow
    >>> res = pipeline.run(number_of_cores=1) # doctest: +SKIP

    """

    bids_dir = None
    output_dir = None
    subject = None
    wf = None
    deltatTV = "0.75"
    lambdaTV = "0.001"
    primal_dual_loops = "20"
    sr_id = 1
    session = None

    m_stacks = None

    # Custom interfaces options
    m_skip_svr = None
    m_skip_nlm_denoising = None
    m_skip_stacks_ordering = None
    m_do_refine_hr_mask = None

    m_masks_derivatives_dir = None
    use_manual_masks = False

    def __init__(self, bids_dir, output_dir, subject, p_stacks=None, sr_id=1,
                 session=None, paramTV=None, p_masks_derivatives_dir=None,
                 p_dict_custom_interfaces = None):
        """Constructor of AnatomicalPipeline class instance."""

        # BIDS processing parameters
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.subject = subject
        self.sr_id = sr_id
        self.session = session
        self.m_stacks = p_stacks

        # (default) sr tv parameters
        if paramTV is None:
            paramTV = dict()
        self.deltatTV = paramTV["deltatTV"] if "deltatTV" in paramTV.keys() else 0.01
        self.lambdaTV = paramTV["lambdaTV"] if "lambdaTV" in paramTV.keys() else 0.75
        self.primal_dual_loops = paramTV["primal_dual_loops"] if "primal_dual_loops" in paramTV.keys() else 10

        # Use manual/custom brain masks
        # If masks directory is not specified use the automated brain extraction method.
        self.m_masks_derivatives_dir = p_masks_derivatives_dir
        self.use_manual_masks = True if self.m_masks_derivatives_dir is not None else False

        # Custom interfaces and default values.
        if p_dict_custom_interfaces is not None:
            self.m_skip_svr = p_dict_custom_interfaces['skip_svr'] if 'skip_svr' in  p_dict_custom_interfaces.keys() else False
            self.m_do_refine_hr_mask = p_dict_custom_interfaces['do_refine_hr_mask'] if 'do_refine_hr_mask' in  p_dict_custom_interfaces.keys() else False
            self.m_skip_nlm_denoising = p_dict_custom_interfaces['skip_nlm_denoising'] if 'skip_nlm_denoising' in  p_dict_custom_interfaces.keys() else False

            self.m_skip_stacks_ordering = p_dict_custom_interfaces['skip_stacks_ordering'] if \
                ((self.m_stacks is not None) and ('skip_stacks_ordering' in p_dict_custom_interfaces.keys())) else False
        else:
            self.m_skip_svr = False
            self.m_do_refine_hr_mask = False
            self.m_skip_nlm_denoising =  False
            self.m_skip_stacks_ordering = False

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
                                       "nipype",
                                       self.subject,
                                       "rec-{}".format(self.sr_id))
            final_res_dir = os.path.join(self.output_dir,
                                         '-'.join(["pymialsrtk", __version__]),
                                         self.subject)
        else:
            wf_base_dir = os.path.join(self.output_dir,
                                       "nipype",
                                       self.subject,
                                       self.session,
                                       "rec-{}".format(self.sr_id))
            final_res_dir = os.path.join(self.output_dir,
                                         '-'.join(["pymialsrtk", __version__]),
                                         self.subject,
                                         self.session)

        # #if self.sr_id is not None:
        # wf_base_dir = os.path.join(wf_base_dir, self.sr_id)

        if not os.path.exists(wf_base_dir):
            os.makedirs(wf_base_dir)
        print("Process directory: {}".format(wf_base_dir))

        # Workflow name cannot begin with a number (oterhwise ValueError)
        pipeline_name = "srr_pipeline"

        self.wf = Workflow(name=pipeline_name,base_dir=wf_base_dir)
        # srr_nipype_dir = os.path.join(self.wf.base_dir, self.wf.name )

        # Initialization (Not sure we can control the name of nipype log)
        if os.path.isfile(os.path.join(wf_base_dir, "pypeline_" + sub_ses + ".log")):
            os.unlink(os.path.join(wf_base_dir, "pypeline_" + sub_ses + ".log"))
            # open(os.path.join(self.output_dir,"pypeline.log"), 'a').close()

        config.update_config({'logging': {'log_directory': os.path.join(wf_base_dir),
                                          'log_to_file': True},
                              'execution': {
                                  'remove_unnecessary_outputs': False,
                                  'stop_on_first_crash': True,
                                  'stop_on_first_rerun': False,
                                  'crashfile_format': "txt",
                                  'write_provenance': False},
                              'monitoring': {'enabled': True}
                              })

        # config.enable_provenance()

        logging.update_logging(config)
        iflogger = logging.getLogger('nipype.interface')

        iflogger.info("**** Processing ****")

        if self.use_manual_masks:
            dg = Node(interface=DataGrabber(outfields=['T2ws', 'masks']), name='data_grabber')

            dg.inputs.base_directory = self.bids_dir
            dg.inputs.template = '*'
            dg.inputs.raise_on_empty = False
            dg.inputs.sort_filelist = True

            dg.inputs.field_template = dict(T2ws=os.path.join(self.subject, 'anat', sub_ses+'*_run-*_T2w.nii.gz'),
                                            masks=os.path.join('derivatives',
                                                               self.m_masks_derivatives_dir,
                                                               self.subject,
                                                               'anat',
                                                               sub_ses+'*_run-*_*mask.nii.gz'))
            if self.session is not None:
                dg.inputs.field_template = dict(T2ws=os.path.join(self.subject,
                                                                  self.session,
                                                                  'anat',
                                                                  '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])),
                                                masks=os.path.join('derivatives',
                                                                   self.m_masks_derivatives_dir,
                                                                   self.subject,
                                                                   self.session,
                                                                   'anat',
                                                                   '_'.join([sub_ses, '*run-*', '*mask.nii.gz'])))
            brainMask = MapNode(interface=IdentityInterface(fields=['out_file']),
                                name='brain_masks_bypass',
                                iterfield=['out_file'])

        else:
            dg = Node(interface=DataGrabber(outfields=['T2ws']), name='data_grabber')

            dg.inputs.base_directory = self.bids_dir
            dg.inputs.template = '*'
            dg.inputs.raise_on_empty = False
            dg.inputs.sort_filelist = True

            dg.inputs.field_template = dict(T2ws=os.path.join(self.subject,
                                                              'anat', sub_ses+'*_run-*_T2w.nii.gz'))
            if self.session is not None:
                dg.inputs.field_template = dict(T2ws=os.path.join(self.subject,
                                                                  self.session, 'anat', '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])))

            if self.m_stacks is not None:
                print('if is not self.m_stacks !!! ')
                t2ws_filter_prior_masks = Node(interface=preprocess.FilteringByRunid(), name='t2ws_filter_prior_masks')
                t2ws_filter_prior_masks.inputs.stacks_id = self.m_stacks

            brainMask = MapNode(interface = preprocess.BrainExtraction(),
                                name='brainExtraction',
                                iterfield=['in_file'])

            brainMask.inputs.bids_dir = self.bids_dir
            brainMask.inputs.in_ckpt_loc = pkg_resources.resource_filename("pymialsrtk",
                                                                           os.path.join("data",
                                                                                        "Network_checkpoints",
                                                                                        "Network_checkpoints_localization",
                                                                                        "Unet.ckpt-88000.index")).split('.index')[0]
            brainMask.inputs.threshold_loc = 0.49
            brainMask.inputs.in_ckpt_seg = pkg_resources.resource_filename("pymialsrtk",
                                                                           os.path.join("data",
                                                                                        "Network_checkpoints",
                                                                                        "Network_checkpoints_segmentation",
                                                                                        "Unet.ckpt-20000.index")).split('.index')[0]
            brainMask.inputs.threshold_seg = 0.5

        t2ws_filtered = Node(interface=preprocess.FilteringByRunid(), name='t2ws_filtered')
        masks_filtered = Node(interface=preprocess.FilteringByRunid(), name='masks_filtered')


        if not self.m_skip_stacks_ordering:
            stacksOrdering = Node(interface=preprocess.StacksOrdering(), name='stackOrdering')
        else:
            stacksOrdering = Node(interface=IdentityInterface(fields=['stacks_order']), name='stackOrdering')
            stacksOrdering.inputs.stacks_order = self.m_stacks

        if not self.m_skip_nlm_denoising:
            nlmDenoise = MapNode(interface=preprocess.BtkNLMDenoising(),
                                 name='nlmDenoise',
                                 iterfield=['in_file', 'in_mask'])
            nlmDenoise.inputs.bids_dir = self.bids_dir

            # Sans le mask le premier correct slice intensity...
            srtkCorrectSliceIntensity01_nlm = MapNode(interface=preprocess.MialsrtkCorrectSliceIntensity(),
                                                      name='srtkCorrectSliceIntensity01_nlm',
                                                      iterfield=['in_file', 'in_mask'])
            srtkCorrectSliceIntensity01_nlm.inputs.bids_dir = self.bids_dir
            srtkCorrectSliceIntensity01_nlm.inputs.out_postfix = '_uni'

        srtkCorrectSliceIntensity01 = MapNode(interface=preprocess.MialsrtkCorrectSliceIntensity(),
                                              name='srtkCorrectSliceIntensity01',
                                                  iterfield=['in_file', 'in_mask'])
        srtkCorrectSliceIntensity01.inputs.bids_dir = self.bids_dir
        srtkCorrectSliceIntensity01.inputs.out_postfix = '_uni'

        srtkSliceBySliceN4BiasFieldCorrection = MapNode(interface=preprocess.MialsrtkSliceBySliceN4BiasFieldCorrection(),
                                                     name='srtkSliceBySliceN4BiasFieldCorrection',
                                                        iterfield=['in_file', 'in_mask'])
        srtkSliceBySliceN4BiasFieldCorrection.inputs.bids_dir = self.bids_dir

        srtkSliceBySliceCorrectBiasField = MapNode(interface=preprocess.MialsrtkSliceBySliceCorrectBiasField(),
                                                   name='srtkSliceBySliceCorrectBiasField',
                                                   iterfield=['in_file', 'in_mask', 'in_field'])
        srtkSliceBySliceCorrectBiasField.inputs.bids_dir = self.bids_dir


    # 4-modules sequence to be defined as a stage.
        if not self.m_skip_nlm_denoising:
            srtkCorrectSliceIntensity02_nlm = MapNode(interface=preprocess.MialsrtkCorrectSliceIntensity(),
                                                      name='srtkCorrectSliceIntensity02_nlm',
                                                      iterfield=['in_file','in_mask'])
            srtkCorrectSliceIntensity02_nlm.inputs.bids_dir = self.bids_dir

            srtkIntensityStandardization01_nlm = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization01_nlm')
            srtkIntensityStandardization01_nlm.inputs.bids_dir = self.bids_dir

            srtkHistogramNormalization_nlm = Node(interface=preprocess.MialsrtkHistogramNormalization(), name='srtkHistogramNormalization_nlm')
            srtkHistogramNormalization_nlm.inputs.bids_dir = self.bids_dir

            srtkIntensityStandardization02_nlm = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization02_nlm')
            srtkIntensityStandardization02_nlm.inputs.bids_dir = self.bids_dir


    # 4-modules sequence to be defined as a stage.
        srtkCorrectSliceIntensity02 = MapNode(interface=preprocess.MialsrtkCorrectSliceIntensity(),
                                          name='srtkCorrectSliceIntensity02',
                                          iterfield=['in_file', 'in_mask'])
        srtkCorrectSliceIntensity02.inputs.bids_dir = self.bids_dir

        srtkIntensityStandardization01 = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization01')
        srtkIntensityStandardization01.inputs.bids_dir = self.bids_dir

        srtkHistogramNormalization = Node(interface=preprocess.MialsrtkHistogramNormalization(), name='srtkHistogramNormalization')
        srtkHistogramNormalization.inputs.bids_dir = self.bids_dir

        srtkIntensityStandardization02 = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization02')
        srtkIntensityStandardization02.inputs.bids_dir = self.bids_dir


        srtkMaskImage01 = MapNode(interface=preprocess.MialsrtkMaskImage(),
                                  name='srtkMaskImage01',
                                  iterfield=['in_file', 'in_mask'])
        srtkMaskImage01.inputs.bids_dir = self.bids_dir

        srtkImageReconstruction = Node(interface=reconstruction.MialsrtkImageReconstruction(), name='srtkImageReconstruction')
        srtkImageReconstruction.inputs.bids_dir = self.bids_dir
        srtkImageReconstruction.inputs.sub_ses = sub_ses
        srtkImageReconstruction.inputs.no_reg = self.m_skip_svr

        srtkTVSuperResolution = Node(interface=reconstruction.MialsrtkTVSuperResolution(), name='srtkTVSuperResolution')
        srtkTVSuperResolution.inputs.bids_dir = self.bids_dir
        srtkTVSuperResolution.inputs.sub_ses = sub_ses
        srtkTVSuperResolution.inputs.in_loop = self.primal_dual_loops
        srtkTVSuperResolution.inputs.in_deltat = self.deltatTV
        srtkTVSuperResolution.inputs.in_lambda = self.lambdaTV
        srtkTVSuperResolution.inputs.use_manual_masks = self.use_manual_masks

        srtkN4BiasFieldCorrection = Node(interface=postprocess.MialsrtkN4BiasFieldCorrection(), name='srtkN4BiasFieldCorrection')
        srtkN4BiasFieldCorrection.inputs.bids_dir = self.bids_dir

        if self.m_do_refine_hr_mask:
            srtkHRMask = Node(interface=postprocess.MialsrtkRefineHRMaskByIntersection(), name='srtkHRMask')
            srtkHRMask.inputs.bids_dir = self.bids_dir
        else:
            srtkHRMask = Node(interface=Function(input_names=["input_image"], output_names=["output_srmask"],
                                    function=postprocess.binarize_image), name='srtkHRMask')

        srtkMaskImage02 = Node(interface=preprocess.MialsrtkMaskImage(), name='srtkMaskImage02')
        srtkMaskImage02.inputs.bids_dir = self.bids_dir

        finalFilenamesGeneration = Node(postprocess.FilenamesGeneration(), name='filenames_gen')
        finalFilenamesGeneration.inputs.sub_ses = sub_ses
        finalFilenamesGeneration.inputs.sr_id = self.sr_id
        finalFilenamesGeneration.inputs.use_manual_masks = self.use_manual_masks

        datasink = Node(DataSink(), name='data_sinker')
        datasink.inputs.base_directory = final_res_dir


        # - Build workflow : connections of the nodes

        # Nodes ready - Linking now
        if self.use_manual_masks:
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
        self.wf.connect(t2ws_filtered, ("output_files", utils.sort_ascending), nlmDenoise, "in_file")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), nlmDenoise, "in_mask")  ## Comment to match docker process

        if not self.m_skip_nlm_denoising:
            self.wf.connect(nlmDenoise, ("out_file", utils.sort_ascending), srtkCorrectSliceIntensity01_nlm, "in_file")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkCorrectSliceIntensity01_nlm, "in_mask")

            self.wf.connect(t2ws_filtered, ("output_files", utils.sort_ascending), srtkCorrectSliceIntensity01, "in_file")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkCorrectSliceIntensity01, "in_mask")

        if not self.m_skip_nlm_denoising:
            self.wf.connect(srtkCorrectSliceIntensity01_nlm, ("out_file", utils.sort_ascending), srtkSliceBySliceN4BiasFieldCorrection, "in_file")
        else:
            self.wf.connect(srtkCorrectSliceIntensity01, ("out_file", utils.sort_ascending),srtkSliceBySliceN4BiasFieldCorrection, "in_file")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkSliceBySliceN4BiasFieldCorrection, "in_mask")

        self.wf.connect(srtkCorrectSliceIntensity01, ("out_file", utils.sort_ascending), srtkSliceBySliceCorrectBiasField, "in_file")
        self.wf.connect(srtkSliceBySliceN4BiasFieldCorrection, ("out_fld_file", utils.sort_ascending), srtkSliceBySliceCorrectBiasField, "in_field")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkSliceBySliceCorrectBiasField, "in_mask")

        if not self.m_skip_nlm_denoising:
            self.wf.connect(srtkSliceBySliceN4BiasFieldCorrection, ("out_im_file", utils.sort_ascending), srtkCorrectSliceIntensity02_nlm, "in_file")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkCorrectSliceIntensity02_nlm, "in_mask")
            self.wf.connect(srtkCorrectSliceIntensity02_nlm, ("out_file", utils.sort_ascending), srtkIntensityStandardization01_nlm, "input_images")
            self.wf.connect(srtkIntensityStandardization01_nlm, ("output_images", utils.sort_ascending), srtkHistogramNormalization_nlm, "input_images")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkHistogramNormalization_nlm, "input_masks")
            self.wf.connect(srtkHistogramNormalization_nlm, ("output_images", utils.sort_ascending), srtkIntensityStandardization02_nlm, "input_images")

        self.wf.connect(srtkSliceBySliceCorrectBiasField, ("out_im_file", utils.sort_ascending), srtkCorrectSliceIntensity02, "in_file")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkCorrectSliceIntensity02, "in_mask")
        self.wf.connect(srtkCorrectSliceIntensity02, ("out_file", utils.sort_ascending), srtkIntensityStandardization01, "input_images")


        self.wf.connect(srtkIntensityStandardization01, ("output_images", utils.sort_ascending), srtkHistogramNormalization, "input_images")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkHistogramNormalization, "input_masks")
        self.wf.connect(srtkHistogramNormalization, ("output_images", utils.sort_ascending), srtkIntensityStandardization02, "input_images")


        if not self.m_skip_nlm_denoising:
            self.wf.connect(srtkIntensityStandardization02_nlm, ("output_images", utils.sort_ascending), srtkMaskImage01, "in_file")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkMaskImage01, "in_mask")
        else:
            self.wf.connect(srtkIntensityStandardization02, ("output_images", utils.sort_ascending), srtkMaskImage01, "in_file")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkMaskImage01, "in_mask")

        self.wf.connect(srtkMaskImage01, "out_im_file", srtkImageReconstruction, "input_images")
        self.wf.connect(masks_filtered, "output_files", srtkImageReconstruction, "input_masks")
        self.wf.connect(stacksOrdering, "stacks_order", srtkImageReconstruction, "stacks_order")

        self.wf.connect(srtkIntensityStandardization02, "output_images", srtkTVSuperResolution, "input_images")
        self.wf.connect(srtkImageReconstruction, ("output_transforms", utils.sort_ascending), srtkTVSuperResolution, "input_transforms")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkTVSuperResolution, "input_masks")
        self.wf.connect(stacksOrdering, "stacks_order", srtkTVSuperResolution, "stacks_order")

        self.wf.connect(srtkImageReconstruction, "output_sdi", srtkTVSuperResolution, "input_sdi")


        if self.m_do_refine_hr_mask:
            self.wf.connect(srtkIntensityStandardization02, ("output_images", utils.sort_ascending), srtkHRMask, "input_images")
            self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), srtkHRMask, "input_masks")
            self.wf.connect(srtkImageReconstruction, ("output_transforms", utils.sort_ascending), srtkHRMask, "input_transforms")
            self.wf.connect(srtkTVSuperResolution, "output_sr", srtkHRMask, "input_sr")
        else:
            self.wf.connect(srtkTVSuperResolution, "output_sr", srtkHRMask, "input_image")

        self.wf.connect(srtkTVSuperResolution, "output_sr", srtkMaskImage02, "in_file")
        self.wf.connect(srtkHRMask, "output_srmask", srtkMaskImage02, "in_mask")


        self.wf.connect(srtkTVSuperResolution, "output_sr", srtkN4BiasFieldCorrection, "input_image")
        self.wf.connect(srtkHRMask, "output_srmask", srtkN4BiasFieldCorrection, "input_mask")

        self.wf.connect(stacksOrdering, "stacks_order", finalFilenamesGeneration, "stacks_order")
        self.wf.connect(finalFilenamesGeneration, "substitutions", datasink, "substitutions")
        self.wf.connect(masks_filtered, ("output_files", utils.sort_ascending), datasink, 'anat.@LRmasks')

        self.wf.connect(srtkIntensityStandardization02, ("output_images", utils.sort_ascending), datasink, 'anat.@LRsPreproc')
        self.wf.connect(srtkMaskImage01, ("out_im_file", utils.sort_ascending), datasink, 'anat.@LRsDenoised')
        self.wf.connect(srtkImageReconstruction, ("output_transforms", utils.sort_ascending), datasink, 'xfm.@transforms')

        self.wf.connect(srtkImageReconstruction, "output_sdi", datasink, 'anat.@SDI')
        self.wf.connect(srtkN4BiasFieldCorrection, "output_image", datasink, 'anat.@SR')
        self.wf.connect(srtkTVSuperResolution, "output_json_path", datasink, 'anat.@SRjson')
        self.wf.connect(srtkHRMask, "output_srmask", datasink, 'anat.@SRmask')

    def run(self, number_of_cores=1):
        """Execute the workflow of the super-resolution reconstruction pipeline.

        Nipype execution engine will take care of the management and execution of
        all processing steps involved in the super-resolution reconstruction pipeline.
        Note that the complete execution graph is saved as a PNG image to support
        transparency on the whole processing.

        Parameters
        ----------
        number_of_cores <int>
            Number of cores / CPUs used by the workflow

        """

        self.wf.write_graph(dotfilename='graph.dot', graph2use='colored', format='png', simple_form=True)
        if number_of_cores > 1:
            res = self.wf.run(plugin='MultiProc', plugin_args={'n_procs': number_of_cores})

        else:
            res = self.wf.run()

        return res
