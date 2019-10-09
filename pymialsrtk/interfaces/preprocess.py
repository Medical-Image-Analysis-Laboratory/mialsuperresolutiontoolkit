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
## NLM denoising  
# 
 

class BtkNLMDenoisingInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_mask = File(desc='Input mask',mandatory=False)
    out_postfix = traits.Str("_nlm", usedefault=True)
    weight = traits.Float(0.1,desc='NLM weight (0.1 by default)')

class BtkNLMDenoisingOutputSpec(TraitedSpec):
    out_file = File(desc='Denoised image')

class BtkNLMDenoising(BaseInterface):

    input_spec = BtkNLMDenoisingInputSpec
    output_spec = BtkNLMDenoisingOutputSpec
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = os.path.join(self.inputs.bids_dir, ''.join((name, self.inputs.out_postfix, ext)))
        print('out_file: {}'.format(out_file))

        if self.inputs.in_mask:
            cmd = 'btkNLMDenoising -i "{}" -m "{}" -o "{}" -b {}'.format(self.inputs.in_file,self.inputs.in_mask,out_file,self.inputs.weight)
        else:
            cmd = 'btkNLMDenoising -i "{}" -o "{}" -b {}'.format(self.inputs.in_file,out_file,self.inputs.weight)
        
        try:
            print('... cmd: {}'.format(cmd))
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        outputs['out_file'] = os.path.join(self.inputs.bids_dir, ''.join((name, self.inputs.out_postfix, ext)))
        return outputs
    
    

class MultipleBtkNLMDenoisingInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be denoised', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be denoised', mandatory = False))
    weight = traits.Float(0.1)
    out_postfix = traits.Str("_nlm", usedefault=True)
    
class MultipleBtkNLMDenoisingOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleBtkNLMDenoising(BaseInterface):
    input_spec = MultipleBtkNLMDenoisingInputSpec
    output_spec = MultipleBtkNLMDenoisingOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.input_images)>0:
            for input_image, input_mask in zip(self.inputs.input_images,self.inputs.input_masks):
                ax = BtkNLMDenoising(bids_dir = self.inputs.bids_dir, in_file = input_image, in_mask = input_mask, out_postfix=self.inputs.out_postfix, weight = self.inputs.weight)
                ax.run()
        else:
            for input_image in self.inputs.input_images:
                ax = BtkNLMDenoising(bids_dir = self.inputs.bids_dir, in_file = input_image, out_postfix=self.inputs.out_postfix, weight = self.inputs.weight)
                ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs


 
# 
## Slice intensity correction 
# 
 
class MialsrtkCorrectSliceIntensityInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_mask = File(desc='Input mask',mandatory=False)
    out_postfix = traits.Str("_csi", usedefault=True)

class MialsrtkCorrectSliceIntensityOutputSpec(TraitedSpec):
    out_file = File(desc='Corrected slice intensities')

    
    
class MialsrtkCorrectSliceIntensity(BaseInterface):
    input_spec = MialsrtkCorrectSliceIntensityInputSpec
    output_spec = MialsrtkCorrectSliceIntensityOutputSpec
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))

        cmd = 'mialsrtkCorrectSliceIntensity "{}" "{}" "{}"'.format(self.inputs.in_file,self.inputs.in_mask,out_file)
        
        try:
            print('... cmd: {}'.format(cmd))
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        outputs['out_file'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))
        return outputs
    
    
    
class MultipleMialsrtkCorrectSliceIntensityInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be corrected for intensity', mandatory = False))
    out_postfix = traits.Str("_csi", usedefault=True)
    
class MultipleMialsrtkCorrectSliceIntensityOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleMialsrtkCorrectSliceIntensity(BaseInterface):
    input_spec = MultipleMialsrtkCorrectSliceIntensityInputSpec
    output_spec = MultipleMialsrtkCorrectSliceIntensityOutputSpec

    def _run_interface(self, runtime):
        for input_image, input_mask in zip(self.inputs.input_images,self.inputs.input_masks):
            ax = MialsrtkCorrectSliceIntensity(bids_dir = self.inputs.bids_dir, in_file = input_image, in_mask = input_mask, out_postfix=self.inputs.out_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs



# 
## Slice by slice N4 bias field correction 
# 

class MialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_mask = File(desc='Input mask',mandatory=True)
    out_im_postfix = traits.Str("_sliceN4corr", usedefault=True)
    out_fld_postfix = traits.Str("_sliceN4field", usedefault=True)

class MialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    out_im_file = File(desc='Corrected slice by slice from N4 bias field')
    out_fld_file = File(desc='slice by slice N4 bias field')

    
    
class MialsrtkSliceBySliceN4BiasFieldCorrection(BaseInterface):
    input_spec = MialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec
    output_spec = MialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_im_postfix, ext)))
        out_fld_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_fld_postfix, ext)))

        cmd = 'mialsrtkSliceBySliceN4BiasFieldCorrection "{}" "{}" "{}" "{}"'.format(self.inputs.in_file, self.inputs.in_mask, out_im_file, out_fld_file)
        
        try:
            print('... cmd: {}'.format(cmd))
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        outputs['out_im_file'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_im_postfix, ext)))
        outputs['out_fld_file'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_fld_postfix, ext)))
        return outputs
    
    
    
class MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be corrected for intensity', mandatory = True))
    out_im_postfix = traits.Str("_sliceN4corr", usedefault=True)
    out_fld_postfix = traits.Str("_sliceN4field", usedefault=True) 
    
class MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())
    output_fields = OutputMultiPath(File())

class MultipleMialsrtkSliceBySliceN4BiasFieldCorrection(BaseInterface):
    input_spec = MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec
    output_spec = MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec

    def _run_interface(self, runtime):
        for input_image, input_mask in zip(self.inputs.input_images,self.inputs.input_masks):
            ax = MialsrtkSliceBySliceN4BiasFieldCorrection(bids_dir = self.inputs.bids_dir, in_file = input_image, in_mask = input_mask, out_im_postfix=self.inputs.out_im_postfix, out_fld_postfix=self.inputs.out_fld_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*_sliceN4corr.nii.gz"))
        outputs['output_fields'] = glob(os.path.abspath("*_sliceN4field.nii.gz"))
        return outputs



# 
## Intensity standardization 
# 

class MialsrtkIntensityStandardizationInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    out_postfix = traits.Str("_ist", usedefault=True)

class MialsrtkIntensityStandardizationOutputSpec(TraitedSpec):
    out_file = File(desc='Intensities standardized')

    
    
class MialsrtkIntensityStandardization(BaseInterface):
    input_spec = MialsrtkIntensityStandardizationInputSpec
    output_spec = MialsrtkIntensityStandardizationOutputSpec
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))

        cmd = 'mialsrtkIntensityStandardization --input "{}" --output "{}"'.format(self.inputs.in_file, out_file)
        
        try:
            print('... cmd: {}'.format(cmd))
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        outputs['out_file'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))
        return outputs
    
    
    
class MultipleMialsrtkIntensityStandardizationInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    out_postfix = traits.Str("_ist", usedefault=True)
    
class MultipleMialsrtkIntensityStandardizationOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleMialsrtkIntensityStandardization(BaseInterface):
    input_spec = MultipleMialsrtkIntensityStandardizationInputSpec
    output_spec = MultipleMialsrtkIntensityStandardizationOutputSpec

    def _run_interface(self, runtime):
        for input_image in self.inputs.input_images:
            print("input_image", input_image)
            ax = MialsrtkIntensityStandardization(bids_dir = self.inputs.bids_dir, in_file = input_image, out_postfix=self.inputs.out_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs