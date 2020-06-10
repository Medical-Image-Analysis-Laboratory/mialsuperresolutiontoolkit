#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import json
import glob

from nipype import config, logging
from nipype.interfaces.io import BIDSDataGrabber,DataGrabber, DataSink, JSONFileSink
#from nipype.pipeline import Node, MapNode, Workflow
from nipype.pipeline import Node, Workflow
from nipype.interfaces.utility import Function

# from pymialsrtk.interfaces.docker import prepareDockerPaths

# Import the implemented interface from pymialsrtk
import pymialsrtk.interfaces.preprocess as preprocess
import pymialsrtk.interfaces.reconstruction as reconstruction
import pymialsrtk.interfaces.postprocess as postprocess

#from traits.api import *
#from nipype.utils.filemanip import split_filename
#from nipype.interfaces.base import traits, isdefined, CommandLine, CommandLineInputSpec,    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec


__version__ = open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), 'version')).read()


    

## Node linkage
## Node linkage
def create_workflow(bids_dir, output_dir, subject, p_stacksOrder, srID, session=None, paramTV={}):

    # (default) sr tv parameters
    deltatTV = paramTV["deltatTV"] if "deltatTV" in paramTV.keys() else 0.01
    lambdaTV = paramTV["lambdaTV"] if "lambdaTV" in paramTV.keys() else 0.75
    primal_dual_loops = paramTV["primal_dual_loops"] if "primal_dual_loops" in paramTV.keys() else 10


    sub_ses = subject
    if session is not None:
        sub_ses = ''.join([sub_ses, '_', session])


    wf_base_dir = os.path.join(output_dir,"nipype", subject, "anatomical_pipeline")
    final_res_dir = os.path.join(bids_dir,'-'.join(["pymialsrtk", __version__]), subject)
    

    if session is not None:
        wf_base_dir = os.path.join(wf_base_dir, session)
        final_res_dir = os.path.join(final_res_dir, session)

    # #if srID is not None:
    # wf_base_dir = os.path.join(wf_base_dir, srID)


    if not os.path.exists(wf_base_dir):
        os.makedirs(wf_base_dir)
    print("Process directory: {}".format(wf_base_dir))

    wf = Workflow(name=srID,base_dir=wf_base_dir)
    # srr_nipype_dir = os.path.join(wf.base_dir, wf.name )
    
    
    # Initialization
    if os.path.isfile(os.path.join(output_dir,"pypeline_"+subject+".log")):
        os.unlink(os.path.join(output_dir,"pypeline_"+subject+".log"))
#         open(os.path.join(output_dir,"pypeline.log"), 'a').close()
        

    config.update_config({'logging': {'log_directory': os.path.join(output_dir), 'log_to_file': True},
                          'execution': {
                              'remove_unnecessary_outputs': False,
                              'stop_on_first_crash': True,
                              'stop_on_first_rerun': False,
                              'crashfile_format': "txt",
                              'write_provenance' : False,},
                          'monitoring': { 'enabled': True }
                        })
    
    logging.update_logging(config)
    iflogger = logging.getLogger('nipype.interface')

    iflogger.info("**** Processing ****")

    
    dg = Node(interface=DataGrabber(outfields = ['T2ws', 'masks']), name='data_grabber')
    
    dg.inputs.base_directory = bids_dir
    dg.inputs.template = '*'
    dg.inputs.raise_on_empty = False
    dg.inputs.sort_filelist=True
    
    dg.inputs.field_template = dict(T2ws=os.path.join(subject, 'anat', sub_ses+'*_run-*_T2w.nii.gz'),
                                   masks=os.path.join('derivatives','manual_masks', subject, 'anat', sub_ses+'*_run-*_*mask.nii.gz'))
    if session is not None:
        dg.inputs.field_template = dict(T2ws=os.path.join( subject, session, 'anat', '_'.join([sub_ses, '*run-*', '*T2w.nii.gz'])),
                                        masks=os.path.join('derivatives','manual_masks', subject, session, 'anat','_'.join([sub_ses, '*run-*', '*mask.nii.gz'])))
    
    
        
    nlmDenoise = Node(interface=preprocess.MultipleBtkNLMDenoising(), name='nlmDenoise')
    nlmDenoise.inputs.bids_dir = bids_dir
    nlmDenoise.inputs.stacksOrder = p_stacksOrder

    
    # Sans le mask le premier correct slice intensity...
    srtkCorrectSliceIntensity01_nlm = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity01_nlm')
    srtkCorrectSliceIntensity01_nlm.inputs.bids_dir = bids_dir
    srtkCorrectSliceIntensity01_nlm.inputs.stacksOrder = p_stacksOrder
    srtkCorrectSliceIntensity01_nlm.inputs.out_postfix = '_uni'

    srtkCorrectSliceIntensity01 = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity01')
    srtkCorrectSliceIntensity01.inputs.bids_dir = bids_dir
    srtkCorrectSliceIntensity01.inputs.stacksOrder = p_stacksOrder
    srtkCorrectSliceIntensity01.inputs.out_postfix = '_uni'

    
    
    srtkSliceBySliceN4BiasFieldCorrection = Node(interface=preprocess.MultipleMialsrtkSliceBySliceN4BiasFieldCorrection(), name='srtkSliceBySliceN4BiasFieldCorrection')
    srtkSliceBySliceN4BiasFieldCorrection.inputs.bids_dir = bids_dir
    srtkSliceBySliceN4BiasFieldCorrection.inputs.stacksOrder = p_stacksOrder
    
    srtkSliceBySliceCorrectBiasField = Node(interface=preprocess.MultipleMialsrtkSliceBySliceCorrectBiasField(), name='srtkSliceBySliceCorrectBiasField')
    srtkSliceBySliceCorrectBiasField.inputs.bids_dir = bids_dir
    srtkSliceBySliceCorrectBiasField.inputs.stacksOrder = p_stacksOrder
    
    
    
    srtkCorrectSliceIntensity02_nlm = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity02_nlm')
    srtkCorrectSliceIntensity02_nlm.inputs.bids_dir = bids_dir
    srtkCorrectSliceIntensity02_nlm.inputs.stacksOrder = p_stacksOrder

    srtkCorrectSliceIntensity02 = Node(interface=preprocess.MultipleMialsrtkCorrectSliceIntensity(), name='srtkCorrectSliceIntensity02')
    srtkCorrectSliceIntensity02.inputs.bids_dir = bids_dir
    srtkCorrectSliceIntensity02.inputs.stacksOrder = p_stacksOrder
    
    
    srtkIntensityStandardization01 = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization01')
    srtkIntensityStandardization01.inputs.bids_dir = bids_dir
    
    
    srtkIntensityStandardization01_nlm = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization01_nlm')
    srtkIntensityStandardization01_nlm.inputs.bids_dir = bids_dir
    
    
    srtkHistogramNormalization = Node(interface=preprocess.MialsrtkHistogramNormalization(), name='srtkHistogramNormalization')
    srtkHistogramNormalization.inputs.bids_dir = bids_dir
    srtkHistogramNormalization.inputs.stacksOrder = p_stacksOrder
    
    srtkHistogramNormalization_nlm = Node(interface=preprocess.MialsrtkHistogramNormalization(), name='srtkHistogramNormalization_nlm')  
    srtkHistogramNormalization_nlm.inputs.bids_dir = bids_dir
    srtkHistogramNormalization_nlm.inputs.stacksOrder = p_stacksOrder
    
    
    srtkIntensityStandardization02 = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization02')
    srtkIntensityStandardization02.inputs.bids_dir = bids_dir
    
    
    srtkIntensityStandardization02_nlm = Node(interface=preprocess.MialsrtkIntensityStandardization(), name='srtkIntensityStandardization02_nlm')
    srtkIntensityStandardization02_nlm.inputs.bids_dir = bids_dir
    
    
    srtkMaskImage01 = Node(interface=preprocess.MultipleMialsrtkMaskImage(), name='srtkMaskImage01')
    srtkMaskImage01.inputs.bids_dir = bids_dir
    srtkMaskImage01.inputs.stacksOrder = p_stacksOrder


    srtkImageReconstruction = Node(interface=reconstruction.MialsrtkImageReconstruction(), name='srtkImageReconstruction')  
    srtkImageReconstruction.inputs.bids_dir = bids_dir
    srtkImageReconstruction.inputs.stacksOrder = p_stacksOrder 

    
    srtkImageReconstruction.inputs.sub_ses = sub_ses
    
    srtkTVSuperResolution = Node(interface=reconstruction.MialsrtkTVSuperResolution(), name='srtkTVSuperResolution')  
    srtkTVSuperResolution.inputs.bids_dir = bids_dir
    srtkTVSuperResolution.inputs.stacksOrder = p_stacksOrder
    srtkTVSuperResolution.inputs.sub_ses = sub_ses
    srtkTVSuperResolution.inputs.in_loop = primal_dual_loops
    srtkTVSuperResolution.inputs.in_deltat = deltatTV
    srtkTVSuperResolution.inputs.in_lambda = lambdaTV
    
    

    srtkRefineHRMaskByIntersection = Node(interface=postprocess.MialsrtkRefineHRMaskByIntersection(), name='srtkRefineHRMaskByIntersection')
    srtkRefineHRMaskByIntersection.inputs.bids_dir = bids_dir
    srtkRefineHRMaskByIntersection.inputs.stacksOrder = p_stacksOrder
    
    srtkN4BiasFieldCorrection = Node(interface=postprocess.MialsrtkN4BiasFieldCorrection(), name='srtkN4BiasFieldCorrection')
    srtkN4BiasFieldCorrection.inputs.bids_dir = bids_dir
    
    
    srtkMaskImage02 = Node(interface=preprocess.MialsrtkMaskImage(), name='srtkMaskImage02')
    srtkMaskImage02.inputs.bids_dir = bids_dir
    


    datasink = Node(DataSink(), name='sinker')
    datasink.inputs.base_directory = final_res_dir
    

    # JSON file SRTV
    output_dict = {}
    output_dict["Description"] = "Isotropic high-resolution image reconstructed using the Total-Variation Super-Resolution algorithm provided by MIALSRTK"
    output_dict["Input sources run order"] = p_stacksOrder
    output_dict["CustomMetaData"] = {}
    output_dict["CustomMetaData"]["Number of scans used"] = str(len(p_stacksOrder))
    output_dict["CustomMetaData"]["TV regularization weight lambda"] = lambdaTV
    output_dict["CustomMetaData"]["Optimization time step"] = deltatTV
    output_dict["CustomMetaData"]["Primal/dual loops"] = primal_dual_loops

    dictsink = JSONFileSink(name='jsonsinker')
    dictsink.inputs.in_dict = output_dict

    dictsink.inputs.out_file = os.path.join(final_res_dir, 'anat', sub_ses+'_rec-SR'+'_id-'+srID+'_T2w.json')  
    

    #
    ## Nodes ready - Linking now
    
    wf.connect(dg, "T2ws", nlmDenoise, "input_images")
#     wf.connect(dg, "masks", nlmDenoise, "input_masks")  ## Comment to match docker process
    
    wf.connect(nlmDenoise, "output_images", srtkCorrectSliceIntensity01_nlm, "input_images")
    wf.connect(dg, "masks", srtkCorrectSliceIntensity01_nlm, "input_masks")
    
    wf.connect(dg, "T2ws", srtkCorrectSliceIntensity01, "input_images")
    wf.connect(dg, "masks", srtkCorrectSliceIntensity01, "input_masks")
    
    wf.connect(srtkCorrectSliceIntensity01_nlm, "output_images", srtkSliceBySliceN4BiasFieldCorrection, "input_images")
    wf.connect(dg, "masks", srtkSliceBySliceN4BiasFieldCorrection, "input_masks")
    
    wf.connect(srtkCorrectSliceIntensity01, "output_images", srtkSliceBySliceCorrectBiasField, "input_images")
    wf.connect(srtkSliceBySliceN4BiasFieldCorrection, "output_fields", srtkSliceBySliceCorrectBiasField, "input_fields")
    wf.connect(dg, "masks", srtkSliceBySliceCorrectBiasField, "input_masks")
    
    wf.connect(srtkSliceBySliceCorrectBiasField, "output_images", srtkCorrectSliceIntensity02, "input_images")
    wf.connect(dg, "masks", srtkCorrectSliceIntensity02, "input_masks")
    
    wf.connect(srtkSliceBySliceN4BiasFieldCorrection, "output_images", srtkCorrectSliceIntensity02_nlm, "input_images")
    wf.connect(dg, "masks", srtkCorrectSliceIntensity02_nlm, "input_masks")
    
    wf.connect(srtkCorrectSliceIntensity02, "output_images", srtkIntensityStandardization01, "input_images")
    
    wf.connect(srtkCorrectSliceIntensity02_nlm, "output_images", srtkIntensityStandardization01_nlm, "input_images")
    
    wf.connect(srtkIntensityStandardization01, "output_images", srtkHistogramNormalization, "input_images")
    wf.connect(dg, "masks", srtkHistogramNormalization, "input_masks")
    
    wf.connect(srtkIntensityStandardization01_nlm, "output_images", srtkHistogramNormalization_nlm, "input_images")
    wf.connect(dg, "masks", srtkHistogramNormalization_nlm, "input_masks")
    
    wf.connect(srtkHistogramNormalization, "output_images", srtkIntensityStandardization02, "input_images")
    wf.connect(srtkHistogramNormalization_nlm, "output_images", srtkIntensityStandardization02_nlm, "input_images")
    
    
    wf.connect(srtkIntensityStandardization02_nlm, "output_images", srtkMaskImage01, "input_images")
    wf.connect(dg, "masks", srtkMaskImage01, "input_masks")
    
    
    wf.connect(srtkMaskImage01, "output_images", srtkImageReconstruction, "input_images")
    wf.connect(dg, "masks", srtkImageReconstruction, "input_masks")
    
    wf.connect(srtkIntensityStandardization02, "output_images", srtkTVSuperResolution, "input_images")
    wf.connect(srtkImageReconstruction, "output_transforms", srtkTVSuperResolution, "input_transforms")
    wf.connect(dg, "masks", srtkTVSuperResolution, "input_masks")
    wf.connect(srtkImageReconstruction, "output_sdi", srtkTVSuperResolution, "input_sdi")
    
    
    wf.connect(srtkIntensityStandardization02, "output_images", srtkRefineHRMaskByIntersection, "input_images")
    wf.connect(dg, "masks", srtkRefineHRMaskByIntersection, "input_masks")
    wf.connect(srtkImageReconstruction, "output_transforms", srtkRefineHRMaskByIntersection, "input_transforms")
    wf.connect(srtkTVSuperResolution, "output_sr", srtkRefineHRMaskByIntersection, "input_sr")
    
    wf.connect(srtkTVSuperResolution, "output_sr", srtkN4BiasFieldCorrection, "input_image")
    wf.connect(srtkRefineHRMaskByIntersection, "output_SRmask", srtkN4BiasFieldCorrection, "input_mask")
    
    wf.connect(srtkTVSuperResolution, "output_sr", srtkMaskImage02, "in_file")
    wf.connect(srtkRefineHRMaskByIntersection, "output_SRmask", srtkMaskImage02, "in_mask")
    
    
    
    #
    ### - Saving files
    
    
    substitutions = []


    for stack in p_stacksOrder:
    
        print( sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm.nii.gz', '    --->     ',sub_ses+'_run-'+str(stack)+'_id-'+srID+'_T2w_preproc.nii.gz')
        substitutions.append( ( sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm.nii.gz', sub_ses+'_run-'+str(stack)+'_id-'+srID+'_T2w_preproc.nii.gz') )
        
        print( sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm_transform_'+str(len(p_stacksOrder))+'V.txt', '    --->     ', sub_ses+'_run-'+str(stack)+'_id-'+srID+'_T2w_from-origin_to-SDI_mode-image_xfm.txt')
        substitutions.append( ( sub_ses+'_run-'+str(stack)+'_T2w_nlm_uni_bcorr_histnorm_transform_'+str(len(p_stacksOrder))+'V.txt', sub_ses+'_run-'+str(stack)+'_id-'+srID+'_T2w_from-origin_to-SDI_mode-image_xfm.txt') )
        
        print( sub_ses+'_run-'+str(stack)+'_T2w_uni_bcorr_histnorm_LRmask.nii.gz', '    --->     ', sub_ses+'_run-'+str(stack)+'_id-'+srID+'_T2w_desc-LRmask.nii.gz')
        substitutions.append( ( sub_ses+'_run-'+str(stack)+'_T2w_uni_bcorr_histnorm_LRmask.nii.gz', sub_ses+'_run-'+str(stack)+'_id-'+srID+'_T2w_desc-LRmask.nii.gz') )

        
    print( 'SDI_'+sub_ses+'_'+str(len(p_stacksOrder))+'V_rad1.nii.gz', '    --->     ', sub_ses+'_rec-SDI'+'_id-'+srID+'_T2w.nii.gz')
    substitutions.append( ( 'SDI_'+sub_ses+'_'+str(len(p_stacksOrder))+'V_rad1.nii.gz', sub_ses+'_rec-SDI'+'_id-'+srID+'_T2w.nii.gz') )

    print( 'SRTV_'+sub_ses+'_'+str(len(p_stacksOrder))+'V_rad1_gbcorr.nii.gz', '    --->     ', sub_ses+'_rec-SR'+'_id-'+srID+'_T2w.nii.gz')
    substitutions.append( ( 'SRTV_'+sub_ses+'_'+str(len(p_stacksOrder))+'V_rad1_gbcorr.nii.gz', sub_ses+'_rec-SR'+'_id-'+srID+'_T2w.nii.gz') )
    
    print( sub_ses+'_T2w_uni_bcorr_histnorm_srMask.nii.gz', '    --->     ', sub_ses+'_rec-SR'+'_id-'+srID+'_T2w_desc-brain_mask.nii.gz')
    substitutions.append( ( sub_ses+'_T2w_uni_bcorr_histnorm_srMask.nii.gz', sub_ses+'_rec-SR'+'_id-'+srID+'_T2w_desc-SRmask.nii.gz') )

    
        
    datasink.inputs.substitutions = substitutions
    
    wf.connect(srtkMaskImage01, "output_images", datasink, 'preproc')
    wf.connect(srtkImageReconstruction, "output_transforms", datasink, 'xfm')
    wf.connect(srtkRefineHRMaskByIntersection, "output_LRmasks", datasink, 'postproc')
    
    wf.connect(srtkImageReconstruction, "output_sdi", datasink, 'anat')
    wf.connect(srtkN4BiasFieldCorrection, "output_image", datasink, 'anat.@SR')
    wf.connect(srtkRefineHRMaskByIntersection, "output_SRmask", datasink, 'postproc.@SRmask')
    
    
    return wf, dictsink



def main(bids_dir, output_dir, subject, p_stacksOrder, session, paramTV={}, number_of_cores=1, srID=None):

    subject = 'sub-'+subject
    if session is not None:
        session = 'ses-'+session

    if srID is None:
        srID = "01" 
    
    wf, dictsink = create_workflow(bids_dir, output_dir, subject, p_stacksOrder, srID, session, paramTV)
    

    if(number_of_cores != 1):
        res = wf.run(plugin='MultiProc', plugin_args={'n_procs' : self.number_of_cores})
        dictsink.run()
    else:
        res = wf.run()
        dictsink.run()


    wf.write_graph()




def get_parser():
    import argparse
    p = argparse.ArgumentParser(description='Entrypoint script to the MIALsrtk pipeline')
    p.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    p.add_argument('output_dir', help='The directory where the output files '
                        'should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')
    p.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                        'Only participant is available',
                        choices=['participant'])
    p.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                       'corresponds to sub-<participant_label> from the BIDS spec '
                       '(so it does not include "sub-"). If this parameter is not '
                       'provided all subjects should be analyzed. Multiple '
                       'participants can be specified with a space separated list.',
                       nargs="+")
    
    p.add_argument('--param_file', help='Path to a JSON file containing subjects\' exams ' 
                       'information and super-resolution total variation parameters.', 
                       default='/fetaldata/code/participants_param.json', type=str)
    #p.add_argument('-v', '--version', action='version',
                        #version='BIDS-App')
    return p


if __name__ == '__main__':

    
    bids_dir = os.path.join('/fetaldata')

    
    parser = get_parser()
    args = parser.parse_args()
    
    print(args.param_file)
    with open(args.param_file, 'r') as f:
        participants_params = json.load(f)
        print(participants_params)
    print()
    

    if len(args.participant_label) >= 1:
        for sub in args.participant_label:
            
            if sub in participants_params.keys():
                sr_list = participants_params[sub]
                
                for iSr, sr_params in enumerate(sr_list):
                    
                    ses = sr_params["session"] if "session" in sr_params.keys() else None

                    print('sr_params')
                    if not "stacksOrder" in sr_params.keys() or not "sr-id" in sr_params.keys():
                        print('Do not process subjects %s because of missing parameters.' % sub)
                        continue

                    if 'paramTV' in sr_params.keys():
                        main(bids_dir=args.bids_dir, 
                            output_dir=args.output_dir, 
                            subject=sub, 
                            p_stacksOrder=sr_params['stacksOrder'], 
                            session=ses, 
                            paramTV=sr_params['paramTV'], 
                            srID=sr_params['sr-id'])
                        
                    else:
                        main(bids_dir=args.bids_dir, 
                            output_dir=args.output_dir, 
                            subject=sub, 
                            p_stacksOrder=sr_params['stacksOrder'], 
                            session=ses, 
                            srID=sr_params['sr-id'])
                        


    else:
        print('ERROR: Processing of all dataset not implemented yet\n At least one participant label should be provided')
