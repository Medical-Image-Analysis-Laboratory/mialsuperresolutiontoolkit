#!/usr/bin/env python
# coding: utf-8

import os

import pkg_resources

from nipype import config, logging
# from nipype.interfaces.io import BIDSDataGrabber
from nipype.interfaces.io import DataGrabber, DataSink, JSONFileSink
# from nipype.pipeline import Node, MapNode, Workflow
from nipype.pipeline import Node, Workflow

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess

# Get pymialsrtk version
from pymialsrtk.info import __version__


class AnatomicalPipeline:
    """
    Description of the class and attributes
    """

    bids_dir = None
    output_dir = None
    subject = None
    wf = None
    dictsink = None
    deltatTV = "0.75"
    lambdaTV = "0.001"
    primal_dual_loops = "20"
    srID = "01"
    session = None
    p_stacksOrder = None
    use_manual_masks = False

    def __init__(self, bids_dir, output_dir, subject,
                 p_stacksOrder, srID, session=None, paramTV=dict(),
                 use_manual_masks=False):
        """
        Constructor for instance of AnatomicalPipeline class
        """

        # BIDS processing parameters
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.subject = subject
        self.srID = srID
        self.session = session
        self.p_stacksOrder = p_stacksOrder

        # (default) sr tv parameters
        self.deltatTV = paramTV["deltatTV"] if "deltatTV" in paramTV.keys() else 0.01
        self.lambdaTV = paramTV["lambdaTV"] if "lambdaTV" in paramTV.keys() else 0.75
        self.primal_dual_loops = paramTV["primal_dual_loops"] if "primal_dual_loops" in paramTV.keys() else 10

        # Use manual/custom brain masks
        # By defaut use the automated brain extraction method
        self.use_manual_masks = use_manual_masks

    def create_workflow(self):
        """
        Create the Niype workflow of the super-resolution pipeline
        """

        sub_ses = self.subject
        if self.session is not None:
            sub_ses = ''.join([sub_ses, '_', self.session])

        wf_base_dir = os.path.join(self.output_dir, "nipype", self.subject, "anatomical_pipeline")
        final_res_dir = os.path.join(self.output_dir, '-'.join(["pymialsrtk", __version__]), self.subject)

        if self.session is not None:
            wf_base_dir = os.path.join(wf_base_dir, self.session)
            final_res_dir = os.path.join(final_res_dir, self.session)

        # #if self.srID is not None:
        # wf_base_dir = os.path.join(wf_base_dir, self.srID)

        if not os.path.exists(wf_base_dir):
            os.makedirs(wf_base_dir)
        print("Process directory: {}".format(wf_base_dir))

        # Workflow name cannot begin with a number (oterhwise ValueError)
        pipeline_name = "rec{}".format(self.srID)

        self.wf = Workflow(name=pipeline_name,base_dir=wf_base_dir)
        # srr_nipype_dir = os.path.join(self.wf.base_dir, self.wf.name )

        # Initialization (Not sure we can control the name of nipype log)
        if os.path.isfile(os.path.join(wf_base_dir, "pypeline_" + self.subject + ".log")):
            os.unlink(os.path.join(wf_base_dir, "pypeline_" + self.subject + ".log"))
            # open(os.path.join(self.output_dir,"pypeline.log"), 'a').close()

        config.update_config({'logging': {'log_directory': os.path.join(wf_base_dir), 'log_to_file': True},
                              'execution': {
                                  'remove_unnecessary_outputs': False,
                                  'stop_on_first_crash': True,
                                  'stop_on_first_rerun': False,
                                  'crashfile_format': "txt",
                                  'write_provenance': False},
                              'monitoring': {'enabled': True}
                              })

        config.enable_provenance()

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
                                                               'manual_masks',
                                                               self.subject,
                                                               'anat',
                                                               sub_ses+'*_run-*_*mask.nii.gz'))
            if self.session is not None:
                dg.inputs.field_template = dict(T2ws=os.path.join(self.subject,
                                                                  self.session,
                                                                  'anat',
                                                                  '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])),
                                                masks=os.path.join('derivatives',
                                                                   'manual_masks',
                                                                   self.subject,
                                                                   self.session,
                                                                   'anat',
                                                                   '_'.join([sub_ses, '*run-*', '*mask.nii.gz'])))
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

            brainMask = Node(interface = preprocess.MultipleBrainExtraction(), name='Multiple_Brain_extraction')
            brainMask.inputs.bids_dir = self.bids_dir
            brainMask.inputs.in_ckpt_loc = pkg_resources.resource_filename("pymialsrtk",
                                                                           "data/Network_checkpoints/Network_checkpoints_localization/Unet.ckpt-88000")
            brainMask.inputs.threshold_loc = 0.49
            brainMask.inputs.in_ckpt_seg = pkg_resources.resource_filename("pymialsrtk",
                                                                           "data/Network_checkpoints/Network_checkpoints_segmentation/Unet.ckpt-20000")
            brainMask.inputs.threshold_seg = 0.5

        nlmDenoise = Node(interface=preprocess.MultipleBtkNLMDenoising(), name='nlmDenoise')
        nlmDenoise.inputs.bids_dir = self.bids_dir
        nlmDenoise.inputs.stacksOrder = self.p_stacksOrder

        # Sans le mask le premier correct slice intensity...
        srtkCorrectSliceIntensity01_nlm = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity01_nlm')
        srtkCorrectSliceIntensity01_nlm.inputs.bids_dir = self.bids_dir
        srtkCorrectSliceIntensity01_nlm.inputs.stacksOrder = self.p_stacksOrder
        srtkCorrectSliceIntensity01_nlm.inputs.out_postfix = '_uni'

        srtkCorrectSliceIntensity01 = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity01')
        srtkCorrectSliceIntensity01.inputs.bids_dir = self.bids_dir
        srtkCorrectSliceIntensity01.inputs.stacksOrder = self.p_stacksOrder
        srtkCorrectSliceIntensity01.inputs.out_postfix = '_uni'

        srtkSliceBySliceN4BiasFieldCorrection = Node(interface=preprocess.MultipleMialsrtkSliceBySliceN4BiasFieldCorrection(), name='srtkSliceBySliceN4BiasFieldCorrection')
        srtkSliceBySliceN4BiasFieldCorrection.inputs.bids_dir = self.bids_dir
        srtkSliceBySliceN4BiasFieldCorrection.inputs.stacksOrder = self.p_stacksOrder

        srtkSliceBySliceCorrectBiasField = Node(interface=preprocess.MultipleMialsrtkSliceBySliceCorrectBiasField(), name='srtkSliceBySliceCorrectBiasField')
        srtkSliceBySliceCorrectBiasField.inputs.bids_dir = self.bids_dir
        srtkSliceBySliceCorrectBiasField.inputs.stacksOrder = self.p_stacksOrder

        srtkCorrectSliceIntensity02_nlm = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity02_nlm')
        srtkCorrectSliceIntensity02_nlm.inputs.bids_dir = self.bids_dir
        srtkCorrectSliceIntensity02_nlm.inputs.stacksOrder = self.p_stacksOrder

        srtkCorrectSliceIntensity02 = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity02')
        srtkCorrectSliceIntensity02.inputs.bids_dir = self.bids_dir
        srtkCorrectSliceIntensity02.inputs.stacksOrder = self.p_stacksOrder

        srtkIntensityStandardization01 = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization01')
        srtkIntensityStandardization01.inputs.bids_dir = self.bids_dir

        srtkIntensityStandardization01_nlm = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization01_nlm')
        srtkIntensityStandardization01_nlm.inputs.bids_dir = self.bids_dir

        srtkHistogramNormalization = Node(interface=preprocess.MialsrtkHistogramNormalization(), name='srtkHistogramNormalization')
        srtkHistogramNormalization.inputs.bids_dir = self.bids_dir
        srtkHistogramNormalization.inputs.stacksOrder = self.p_stacksOrder

        srtkHistogramNormalization_nlm = Node(interface=preprocess.MialsrtkHistogramNormalization(), name='srtkHistogramNormalization_nlm')
        srtkHistogramNormalization_nlm.inputs.bids_dir = self.bids_dir
        srtkHistogramNormalization_nlm.inputs.stacksOrder = self.p_stacksOrder

        srtkIntensityStandardization02 = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization02')
        srtkIntensityStandardization02.inputs.bids_dir = self.bids_dir

        srtkIntensityStandardization02_nlm = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization02_nlm')
        srtkIntensityStandardization02_nlm.inputs.bids_dir = self.bids_dir

        srtkMaskImage01 = Node(interface=preprocess.MultipleMialsrtkMaskImage(), name='srtkMaskImage01')
        srtkMaskImage01.inputs.bids_dir = self.bids_dir
        srtkMaskImage01.inputs.stacksOrder = self.p_stacksOrder

        srtkImageReconstruction = Node(interface=reconstruction.MialsrtkImageReconstruction(), name='srtkImageReconstruction')
        srtkImageReconstruction.inputs.bids_dir = self.bids_dir
        srtkImageReconstruction.inputs.stacksOrder = self.p_stacksOrder

        srtkImageReconstruction.inputs.sub_ses = sub_ses

        srtkTVSuperResolution = Node(interface=reconstruction.MialsrtkTVSuperResolution(), name='srtkTVSuperResolution')
        srtkTVSuperResolution.inputs.bids_dir = self.bids_dir
        srtkTVSuperResolution.inputs.stacksOrder = self.p_stacksOrder
        srtkTVSuperResolution.inputs.sub_ses = sub_ses
        srtkTVSuperResolution.inputs.in_loop = self.primal_dual_loops
        srtkTVSuperResolution.inputs.in_deltat = self.deltatTV
        srtkTVSuperResolution.inputs.in_lambda = self.lambdaTV

        srtkRefineHRMaskByIntersection = Node(interface=postprocess.MialsrtkRefineHRMaskByIntersection(), name='srtkRefineHRMaskByIntersection')
        srtkRefineHRMaskByIntersection.inputs.bids_dir = self.bids_dir
        srtkRefineHRMaskByIntersection.inputs.stacksOrder = self.p_stacksOrder

        srtkN4BiasFieldCorrection = Node(interface=postprocess.MialsrtkN4BiasFieldCorrection(), name='srtkN4BiasFieldCorrection')
        srtkN4BiasFieldCorrection.inputs.bids_dir = self.bids_dir

        srtkMaskImage02 = Node(interface=preprocess.MialsrtkMaskImage(), name='srtkMaskImage02')
        srtkMaskImage02.inputs.bids_dir = self.bids_dir

        datasink = Node(DataSink(), name='data_sinker')
        datasink.inputs.base_directory = final_res_dir

        # JSON file SRTV
        output_dict = {}
        output_dict["Description"] = "Isotropic high-resolution image reconstructed using the Total-Variation Super-Resolution algorithm provided by MIALSRTK"
        output_dict["Input sources run order"] = self.p_stacksOrder
        output_dict["CustomMetaData"] = {}
        output_dict["CustomMetaData"]["Number of scans used"] = str(len(self.p_stacksOrder))
        output_dict["CustomMetaData"]["TV regularization weight lambda"] = self.lambdaTV
        output_dict["CustomMetaData"]["Optimization time step"] = self.deltatTV
        output_dict["CustomMetaData"]["Primal/dual loops"] = self.primal_dual_loops

        self.dictsink = JSONFileSink(name='json_sinker')
        self.dictsink.inputs.in_dict = output_dict

        self.dictsink.inputs.out_file = os.path.join(final_res_dir, 'anat', sub_ses+'_rec-SR'+'_id-'+str(self.srID)+'_T2w.json')

        # Nodes ready - Linking now
        if not self.use_manual_masks:
            self.wf.connect(dg, "T2ws", brainMask, "input_images")

        self.wf.connect(dg, "T2ws", nlmDenoise, "input_images")
        # self.wf.connect(dg, "masks", nlmDenoise, "input_masks")  ## Comment to match docker process

        self.wf.connect(nlmDenoise, "output_images", srtkCorrectSliceIntensity01_nlm, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkCorrectSliceIntensity01_nlm, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkCorrectSliceIntensity01_nlm, "input_masks")

        self.wf.connect(dg, "T2ws", srtkCorrectSliceIntensity01, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkCorrectSliceIntensity01, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkCorrectSliceIntensity01, "input_masks")

        self.wf.connect(srtkCorrectSliceIntensity01_nlm, "output_images", srtkSliceBySliceN4BiasFieldCorrection, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkSliceBySliceN4BiasFieldCorrection, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkSliceBySliceN4BiasFieldCorrection, "input_masks")

        self.wf.connect(srtkCorrectSliceIntensity01, "output_images", srtkSliceBySliceCorrectBiasField, "input_images")
        self.wf.connect(srtkSliceBySliceN4BiasFieldCorrection, "output_fields", srtkSliceBySliceCorrectBiasField, "input_fields")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkSliceBySliceCorrectBiasField, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkSliceBySliceCorrectBiasField, "input_masks")
        self.wf.connect(srtkSliceBySliceCorrectBiasField, "output_images", srtkCorrectSliceIntensity02, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkCorrectSliceIntensity02, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkCorrectSliceIntensity02, "input_masks")

        self.wf.connect(srtkSliceBySliceN4BiasFieldCorrection, "output_images", srtkCorrectSliceIntensity02_nlm, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkCorrectSliceIntensity02_nlm, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkCorrectSliceIntensity02_nlm, "input_masks")
        self.wf.connect(srtkCorrectSliceIntensity02, "output_images", srtkIntensityStandardization01, "input_images")

        self.wf.connect(srtkCorrectSliceIntensity02_nlm, "output_images", srtkIntensityStandardization01_nlm, "input_images")

        self.wf.connect(srtkIntensityStandardization01, "output_images", srtkHistogramNormalization, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkHistogramNormalization, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkHistogramNormalization, "input_masks")
        self.wf.connect(srtkIntensityStandardization01_nlm, "output_images", srtkHistogramNormalization_nlm, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkHistogramNormalization_nlm, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkHistogramNormalization_nlm, "input_masks")
        self.wf.connect(srtkHistogramNormalization, "output_images", srtkIntensityStandardization02, "input_images")
        self.wf.connect(srtkHistogramNormalization_nlm, "output_images", srtkIntensityStandardization02_nlm, "input_images")

        self.wf.connect(srtkIntensityStandardization02_nlm, "output_images", srtkMaskImage01, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkMaskImage01, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkMaskImage01, "input_masks")

        self.wf.connect(srtkMaskImage01, "output_images", srtkImageReconstruction, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkImageReconstruction, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkImageReconstruction, "input_masks")

        self.wf.connect(srtkIntensityStandardization02, "output_images", srtkTVSuperResolution, "input_images")
        self.wf.connect(srtkImageReconstruction, "output_transforms", srtkTVSuperResolution, "input_transforms")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkTVSuperResolution, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkTVSuperResolution, "input_masks")
        self.wf.connect(srtkImageReconstruction, "output_sdi", srtkTVSuperResolution, "input_sdi")

        self.wf.connect(srtkIntensityStandardization02, "output_images", srtkRefineHRMaskByIntersection, "input_images")
        if self.use_manual_masks:
            self.wf.connect(dg, "masks", srtkRefineHRMaskByIntersection, "input_masks")
        else:
            self.wf.connect(brainMask, "masks", srtkRefineHRMaskByIntersection, "input_masks")
        self.wf.connect(srtkImageReconstruction, "output_transforms", srtkRefineHRMaskByIntersection, "input_transforms")
        self.wf.connect(srtkTVSuperResolution, "output_sr", srtkRefineHRMaskByIntersection, "input_sr")

        self.wf.connect(srtkTVSuperResolution, "output_sr", srtkN4BiasFieldCorrection, "input_image")
        self.wf.connect(srtkRefineHRMaskByIntersection, "output_SRmask", srtkN4BiasFieldCorrection, "input_mask")

        self.wf.connect(srtkTVSuperResolution, "output_sr", srtkMaskImage02, "in_file")
        self.wf.connect(srtkRefineHRMaskByIntersection, "output_SRmask", srtkMaskImage02, "in_mask")

        # Saving files
        substitutions = []

        for stack in self.p_stacksOrder:
            print(sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm.nii.gz',
                  '    --->     ',
                  sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_desc-preprocSDI_T2w.nii.gz')
            substitutions.append((sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm.nii.gz',
                                 sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_desc-preprocSDI_T2w.nii.gz'))

            if not self.use_manual_masks:
                print(sub_ses+'_run-'+str(stack)+'_T2w_brainMask.nii.gz',
                      '    --->     ',
                      sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_desc-brain_mask.nii.gz')
                substitutions.append((sub_ses+'_run-'+str(stack)+'_T2w_brainMask.nii.gz',
                                     sub_ses+'_run-'+str(stack)+'_desc-brain_mask.nii.gz'))

            print(sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm.nii.gz',
                  '    --->     ',
                  sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_desc-preprocSR_T2w.nii.gz')
            substitutions.append((sub_ses+'_run-'+str(stack)+'_T2w_uni_bcorr_histnorm.nii.gz',
                                 sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_desc-preprocSR_T2w.nii.gz'))

            print(sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm_transform_'+str(len(self.p_stacksOrder))+'V.txt',
                  '    --->     ',
                  sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_T2w_from-origin_to-SDI_mode-image_xfm.txt')
            substitutions.append((sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm_transform_'+str(len(self.p_stacksOrder))+'V.txt',
                                 sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_T2w_from-origin_to-SDI_mode-image_xfm.txt'))

            print(sub_ses+'_run-'+str(stack)+'_T2w_uni_bcorr_histnorm_LRmask.nii.gz',
                  '    --->     ',
                  sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_T2w_desc-brain_mask.nii.gz')
            substitutions.append((sub_ses+'_run-'+str(stack)+'_T2w_uni_bcorr_histnorm_LRmask.nii.gz',
                                 sub_ses+'_run-'+str(stack)+'_id-'+str(self.srID)+'_T2w_desc-brain_mask.nii.gz'))

        print('SDI_'+sub_ses+'_'+str(len(self.p_stacksOrder))+'V_rad1.nii.gz',
              '    --->     ',
              sub_ses+'_rec-SDI'+'_id-'+str(self.srID)+'_T2w.nii.gz')
        substitutions.append(('SDI_'+sub_ses+'_'+str(len(self.p_stacksOrder))+'V_rad1.nii.gz',
                             sub_ses+'_rec-SDI'+'_id-'+str(self.srID)+'_T2w.nii.gz'))

        print('SRTV_'+sub_ses+'_'+str(len(self.p_stacksOrder))+'V_rad1_gbcorr.nii.gz',
              '    --->     ',
              sub_ses+'_rec-SR'+'_id-'+str(self.srID)+'_T2w.nii.gz')
        substitutions.append(('SRTV_'+sub_ses+'_'+str(len(self.p_stacksOrder))+'V_rad1_gbcorr.nii.gz',
                             sub_ses+'_rec-SR'+'_id-'+str(self.srID)+'_T2w.nii.gz'))

        print(sub_ses+'_T2w_uni_bcorr_histnorm_srMask.nii.gz',
              '    --->     ',
              sub_ses+'_rec-SR'+'_id-'+str(self.srID)+'_T2w_desc-brain_mask.nii.gz')
        substitutions.append((sub_ses+'_T2w_uni_bcorr_histnorm_srMask.nii.gz',
                             sub_ses+'_rec-SR'+'_id-'+str(self.srID)+'_T2w_desc-brain_mask.nii.gz'))

        datasink.inputs.substitutions = substitutions

        if not self.use_manual_masks:
            self.wf.connect(brainMask, "masks", datasink, 'anat.@LRmasks')

        self.wf.connect(srtkIntensityStandardization02, "output_images", datasink, 'anat.@LRsPreproc')
        self.wf.connect(srtkMaskImage01, "output_images", datasink, 'anat.@LRsDenoised')
        self.wf.connect(srtkImageReconstruction, "output_transforms", datasink, 'xfm.@transforms')

        self.wf.connect(srtkImageReconstruction, "output_sdi", datasink, 'anat.@SDI')
        self.wf.connect(srtkN4BiasFieldCorrection, "output_image", datasink, 'anat.@SR')
        self.wf.connect(srtkRefineHRMaskByIntersection, "output_SRmask", datasink, 'anat.@SRmask')

    def run(self, number_of_cores=1):
        """
        Execute the super-resolution pipeline and the execution graph
        as a PNG image
        """
        if(number_of_cores != 1):
            res = self.wf.run(plugin='MultiProc', plugin_args={'n_procs': self.number_of_cores})
            self.dictsink.run()
        else:
            res = self.wf.run()
            self.dictsink.run()

        self.wf.write_graph(dotfilename='graph.dot', graph2use='colored', format='png', simple_form=True)
        return res
