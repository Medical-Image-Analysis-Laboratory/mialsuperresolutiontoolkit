# Copyright © 2016-2019 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

""" PyMIALSRTK preprocessing functions
"""

import os

from glob import glob

from traits.api import *

import nibabel as nib

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, isdefined, CommandLine, CommandLineInputSpec,\
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

from pymialsrtk.interfaces.utils import run




# 
## Image Reconstruction
# 

class MialsrtkImageReconstructionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    
    in_roi = traits.Enum(None, "all", "box", "mask", mandatory = True, default='mask', usedefault=True)
    

    input_masks = InputMultiPath(File(desc='')) # , mandatory = True))
    input_images = InputMultiPath(File(desc='')) # , mandatory = True))

    input_rad_dilatation = traits.Float(desc='', default=1, usedefault=True) # , mandatory = True))

    sub_ses = traits.Str("x", usedefault=True)

    out_sdi_prefix = traits.Str("SDI_", usedefault=True)
    out_sdi_postfix = traits.Str("_SDI", usedefault=True)

    out_transf_postfix = traits.Str("_transform", usedefault=True)
    stacksOrder = traits.List(mandatory=False)
    

    in_iter = traits.Int(usedefault=False)
    # in_deblurring = traits.Bool(False, usedefault=True)
    # in_reg = traits.Bool(True, usedefault=True)
    # in_3d = traits.Bool(False, usedefault=True)
    
    # in_margin = traits.Float(usedefault=False)
    # in_epsilon = traits.Float(usedefault=False)
    
    # in_combinedMasks = traits.Str(usedefault=False) ## ?? TODO
    # # in_reference = File(desc='Reference image') # , mandatory=True)

    # in_imresampled = InputMultiPath(File(desc='')) # , mandatory = True))
    # in_imroi = InputMultiPath(File(desc='')) # , mandatory = True))
    

    
class MialsrtkImageReconstructionOutputSpec(TraitedSpec):
    output_sdi = File()
    output_transforms = OutputMultiPath(File(desc='SDI')) 

class MialsrtkImageReconstruction(BaseInterface):
    input_spec = MialsrtkImageReconstructionInputSpec
    output_spec = MialsrtkImageReconstructionOutputSpec

    def _run_interface(self, runtime):
        
        params = []
        params.append(''.join(["--", self.inputs.in_roi]))

        run_nb_images  = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        run_nb_masks  = []
        for in_mask in self.inputs.input_masks:
            cut_avt = in_mask.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_masks.append(int(cut_apr))

        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)

            _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[index_img]))
            transf_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join([name, self.inputs.out_transf_postfix, '_', str(len(self.inputs.stacksOrder)),'V', '.txt']))

            params.append("--input")
            params.append(self.inputs.input_images[index_img])

            params.append("--transform")
            params.append(transf_file)


            if self.inputs.in_roi == "mask":

                index_mask = run_nb_masks.index(order)

                params.append("-m")
                params.append(self.inputs.input_masks[index_mask])


        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([self.inputs.out_sdi_prefix, self.inputs.sub_ses, '_', str(len(self.inputs.stacksOrder)),'V_rad', str(int(self.inputs.input_rad_dilatation)), ext])))        
        #out_file = ''.join(list(out_file))

        params.append("--output")
        params.append(out_file) 
        
        if self.inputs.in_iter:
            params.append("--iter")
            params.append(str(self.inputs.in_iter))


        # if self.inputs.in_imresampled:
        #     for ir in self.inputs.in_imresampled:
        #         params.append("--ir")
        #         params.append(ir)

        # if self.inputs.in_imroi:
        #     for roi in self.inputs.in_imroi:
        #         params.append("--roi")
        #         params.append(roi)


        # if self.inputs.in_deblurring:
        #     params.append("--deblurring")

        # if not self.inputs.in_reg:
        #     params.append("--noreg")

        # if self.inputs.in_3d:
        #     params.append("--3D")

        # if self.inputs.in_margin:
        #     params.append("--margin")
        #     params.append(str(self.inputs.in_margin))

        # if self.inputs.in_epsilon:
        #     params.append("--epsilon")
        #     params.append(str(self.inputs.in_epsilon))

        # if self.inputs.in_iter:
        #     params.append("--iter")
        #     params.append(str(self.inputs.in_iter))

        # if self.inputs.in_combinedMasks:
        #     params.append("--combinedMasks")
        #     params.append(str(self.inputs.in_combinedMasks))

        # if self.inputs.in_reference:
        #     params.append("--reference")
        #     params.append(str(self.inputs.in_reference))
        
        
        cmd = ["mialsrtkImageReconstruction"] 
        cmd += params
        
#         cmd = ["mialsrtkImageReconstruction", "--help"]
        
        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            print("")
            print(cmd)
            print("")
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime
            
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_transforms'] = glob(os.path.abspath("*.txt"))

        _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[0]))
        
        #outputs['output_sdi'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([str('_'), str(self.inputs.out_sdi_prefix), str(len(self.inputs.stacksOrder)),str('V_rad'), str(int(self.inputs.input_rad_dilatation)), ext])))
        outputs['output_sdi'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([self.inputs.out_sdi_prefix, self.inputs.sub_ses, '_', str(len(self.inputs.stacksOrder)),'V_rad', str(int(self.inputs.input_rad_dilatation)), ext])))

        return outputs





#
##  Total Variation Super Resolution
# 

    
class MialsrtkTVSuperResolutionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be SR', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be SR', mandatory = True))
    input_transforms = InputMultiPath(File(desc='', mandatory = True))
    input_sdi = File(File(desc='', mandatory = True))
    deblurring = traits.Bool(False, usedefault=True)
    
    out_prefix = traits.Str("SRTV_", usedefault=True)
    out_srtv_postfix = traits.Str("_gbcorr", usedefault=True)
    out_fld_postfix = traits.Str("_gbcorr", usedefault=True)
    stacksOrder = traits.List(mandatory = False)

    sub_ses = traits.Str("x", usedefault=True)
    
class MialsrtkTVSuperResolutionOutputSpec(TraitedSpec):
    output_sr = File()

class MialsrtkTVSuperResolution(BaseInterface):
    input_spec = MialsrtkTVSuperResolutionInputSpec
    output_spec = MialsrtkTVSuperResolutionOutputSpec

    def _run_interface(self, runtime):

        cmd = ['mialsrtkTVSuperResolution']

        run_nb_images  = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        run_nb_masks  = []
        for in_mask in self.inputs.input_masks:
            cut_avt = in_mask.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_masks.append(int(cut_apr))

        run_nb_transforms  = []
        for in_transform in self.inputs.input_transforms:
            cut_avt = in_transform.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_transforms.append(int(cut_apr))



        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)
            index_mask = run_nb_masks.index(order)
            index_tranform = run_nb_transforms.index(order)

            cmd += ['-i', self.inputs.input_images[index_img]]
            cmd += ['-m', self.inputs.input_masks[index_mask]]
            cmd += ['-t', self.inputs.input_transforms[index_tranform]]



        _, name, ext = split_filename(os.path.abspath(self.inputs.input_sdi))
        name = name.replace('SDI_', self.inputs.out_prefix)
        # out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([name, self.inputs.out_srtv_postfix, ext])))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([self.inputs.out_prefix, self.inputs.sub_ses, '_', str(len(self.inputs.stacksOrder)),'V_rad', self.inputs.out_srtv_postfix, ext])))        

        cmd += ['-r', self.inputs.input_sdi]
        cmd += ['-o', out_file]

        if self.inputs.deblurring:
            cmd += ['--debluring']

        cmd += ['--bregman-loop', '1']
        cmd += ['--loop', str(10)]
        cmd += ['--deltat', str(0.01)]
        cmd += ['--lambda', str(0.75)]

        cmd += ['--iter', '50']
        cmd += ['--step-scale', '10']
        cmd += ['--gamma', '50']
        cmd += ['--inner-thresh', '0.00001']
        cmd += ['--outer-thresh', '0.000001']


        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_sdi))
        name = name.replace('SDI_', self.inputs.out_prefix)
        outputs['output_sr'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([name, self.inputs.out_srtv_postfix, ext])))
        outputs['output_sr'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join(([self.inputs.out_prefix, self.inputs.sub_ses, '_', str(len(self.inputs.stacksOrder)),'V_rad', self.inputs.out_srtv_postfix, ext])))        
        
        return outputs

