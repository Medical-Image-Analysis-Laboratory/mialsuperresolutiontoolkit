# Copyright © 2016-2019 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

""" PyMIALSRTK preprocessing functions
"""

import os
import sys

from glob import glob

import math
import nibabel as nib

import cv2

from medpy.io import load

import scipy.ndimage as snd
from skimage import morphology
from scipy.signal import argrelextrema

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not available. Can not run brain extraction")

try:
    import tflearn
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
except ImportError:
    print("tflearn not available. Can not run brain extraction")

import numpy as np

from traits.api import *

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
    weight = traits.Float(0.1,desc='NLM weight (0.1 by default)', usedefault=True)

class BtkNLMDenoisingOutputSpec(TraitedSpec):
    out_file = File(desc='Denoised image')

class BtkNLMDenoising(BaseInterface):

    input_spec = BtkNLMDenoisingInputSpec
    output_spec = BtkNLMDenoisingOutputSpec 
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))


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
    weight = traits.Float(0.1,desc='NLM weight (0.1 by default)', usedefault=True)
    out_postfix = traits.Str("_nlm", usedefault=True)
    stacksOrder = traits.List(mandatory=False)
    
class MultipleBtkNLMDenoisingOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleBtkNLMDenoising(BaseInterface):
    input_spec = MultipleBtkNLMDenoisingInputSpec
    output_spec = MultipleBtkNLMDenoisingOutputSpec

    def _run_interface(self, runtime):

        run_nb_images  = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        if self.inputs.input_masks:
            run_nb_masks  = []
            for in_mask in self.inputs.input_masks:
                cut_avt = in_mask.split('run-')[1]
                cut_apr = cut_avt.split('_')[0]
                run_nb_masks.append(int(cut_apr))


        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)
                        
            if len(self.inputs.input_masks)>0:
                index_mask = run_nb_masks.index(order)
                ax = BtkNLMDenoising(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], in_mask = self.inputs.input_masks[index_mask], out_postfix=self.inputs.out_postfix, weight = self.inputs.weight)
            else:
                ax = BtkNLMDenoising(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], out_postfix=self.inputs.out_postfix, weight = self.inputs.weight)

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
    out_postfix = traits.Str("", usedefault=True)

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
    out_postfix = traits.Str("", usedefault=True)
    stacksOrder = traits.List(madatory=False)
    
class MultipleMialsrtkCorrectSliceIntensityOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleMialsrtkCorrectSliceIntensity(BaseInterface):
    input_spec = MultipleMialsrtkCorrectSliceIntensityInputSpec
    output_spec = MultipleMialsrtkCorrectSliceIntensityOutputSpec

    def _run_interface(self, runtime):

        run_nb_images  = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        if self.inputs.input_masks:
            run_nb_masks  = []
            for in_mask in self.inputs.input_masks:
                cut_avt = in_mask.split('run-')[1]
                cut_apr = cut_avt.split('_')[0]
                run_nb_masks.append(int(cut_apr))


        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)
                        
            if len(self.inputs.input_masks)>0:
                index_mask = run_nb_masks.index(order)
                ax = MialsrtkCorrectSliceIntensity(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], in_mask = self.inputs.input_masks[index_mask], out_postfix=self.inputs.out_postfix)
            else:
                ax = MialsrtkCorrectSliceIntensity(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], out_postfix=self.inputs.out_postfix)
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
    out_im_postfix = traits.Str("_bcorr", usedefault=True)
    out_fld_postfix = traits.Str("_n4bias", usedefault=True)

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
        if "_uni" in out_fld_file:
            out_fld_file.replace('_uni','')
        

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


        out_fld_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_fld_postfix, ext)))
        if "_uni" in out_fld_file:
            out_fld_file.replace('_uni','')
        outputs['out_fld_file'] = out_fld_file
        return outputs
    
    
    
class MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be corrected for intensity', mandatory = True))
    out_im_postfix = traits.Str("_bcorr", usedefault=True)
    out_fld_postfix = traits.Str("_n4bias", usedefault=True) 
    stacksOrder = traits.List(madatory=False) 
    
class MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())
    output_fields = OutputMultiPath(File())

class MultipleMialsrtkSliceBySliceN4BiasFieldCorrection(BaseInterface):
    input_spec = MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec
    output_spec = MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec

    def _run_interface(self, runtime):

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
            index_mask = run_nb_masks.index(order)
            
            ax = MialsrtkSliceBySliceN4BiasFieldCorrection(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], in_mask = self.inputs.input_masks[index_mask], out_im_postfix=self.inputs.out_im_postfix, out_fld_postfix=self.inputs.out_fld_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath(''.join(["*", self.inputs.out_im_postfix, ".nii.gz"])))
        outputs['output_fields'] = glob(os.path.abspath(''.join(["*", self.inputs.out_fld_postfix, ".nii.gz"])))
        return outputs



# 
## slice by slice correct bias field 
# 


class MialsrtkSliceBySliceCorrectBiasFieldInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_mask = File(desc='Input mask',mandatory=True)
    in_field = File(desc='Input bias field',mandatory=True)
    out_im_postfix = traits.Str("_bcorr", usedefault=True)

class MialsrtkSliceBySliceCorrectBiasFieldOutputSpec(TraitedSpec):
    out_im_file = File(desc='Bias field corrected image')

    
class MialsrtkSliceBySliceCorrectBiasField(BaseInterface):
    input_spec = MialsrtkSliceBySliceCorrectBiasFieldInputSpec
    output_spec = MialsrtkSliceBySliceCorrectBiasFieldOutputSpec
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_im_postfix, ext)))
        

        cmd = 'mialsrtkSliceBySliceCorrectBiasField "{}" "{}" "{}" "{}"'.format(self.inputs.in_file, self.inputs.in_mask, self.inputs.in_field, out_im_file)
        
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
        return outputs
    
    
    
class MultipleMialsrtkSliceBySliceCorrectBiasFieldInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be corrected for intensity', mandatory = True))
    input_fields = InputMultiPath(File(desc='field to remove', mandatory = True))
    out_im_postfix = traits.Str("_bcorr", usedefault=True)
    stacksOrder = traits.List(mandatory=False)
    
class MultipleMialsrtkSliceBySliceCorrectBiasFieldOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleMialsrtkSliceBySliceCorrectBiasField(BaseInterface):
    input_spec = MultipleMialsrtkSliceBySliceCorrectBiasFieldInputSpec
    output_spec = MultipleMialsrtkSliceBySliceCorrectBiasFieldOutputSpec

    def _run_interface(self, runtime):

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

        run_nb_fields  = []
        for in_mask in self.inputs.input_fields:
            cut_avt = in_mask.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_fields.append(int(cut_apr))
        
        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)
            index_mask = run_nb_masks.index(order)
            index_fld = run_nb_fields.index(order)
            ax = MialsrtkSliceBySliceCorrectBiasField(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], in_mask = self.inputs.input_masks[index_mask], in_field=self.inputs.input_fields[index_fld], out_im_postfix=self.inputs.out_im_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath(''.join(["*", self.inputs.out_im_postfix, ".nii.gz"])))
        return outputs




# 
## Intensity standardization 
# 
       
class MialsrtkIntensityStandardizationInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    out_postfix = traits.Str("", usedefault=True)
    in_max = traits.Float(usedefault=False)
    stacksOrder = traits.List(mandatory=False)
    
class MialsrtkIntensityStandardizationOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MialsrtkIntensityStandardization(BaseInterface):
    input_spec = MialsrtkIntensityStandardizationInputSpec
    output_spec = MialsrtkIntensityStandardizationOutputSpec

    def _run_interface(self, runtime):

        cmd = 'mialsrtkIntensityStandardization'
        for input_image in self.inputs.input_images:
            _, name, ext = split_filename(os.path.abspath(input_image))
            out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))
            cmd = cmd + ' --input "{}" --output "{}"'.format(input_image, out_file)

        if self.inputs.in_max:
            cmd = cmd + ' --max "{}"'.format(self.inputs.in_max)
        
        try:
            print('... cmd: {}'.format(cmd))
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime


    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs



# 
## Histogram normalization 
# 


class MialsrtkHistogramNormalizationInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be HistNorm', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be HistNorm', mandatory = False))
    out_postfix = traits.Str("_histnorm", usedefault=True)
    stacksOrder = traits.List(mandatory=False)
    
class MialsrtkHistogramNormalizationOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MialsrtkHistogramNormalization(BaseInterface):
    input_spec = MialsrtkHistogramNormalizationInputSpec
    output_spec = MialsrtkHistogramNormalizationOutputSpec

    def _run_interface(self, runtime):

        cmd = 'python /usr/local/bin/mialsrtkHistogramNormalization.py '


        run_nb_images  = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        if self.inputs.input_masks:
            run_nb_masks  = []
            for in_mask in self.inputs.input_masks:
                cut_avt = in_mask.split('run-')[1]
                cut_apr = cut_avt.split('_')[0]
                run_nb_masks.append(int(cut_apr))


        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)
            _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[index_img]))
            out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))
            
            if len(self.inputs.input_masks)>0:
                index_mask = run_nb_masks.index(order)
                cmd = cmd + ' -i "{}" -o "{}" -m "{}" '.format(self.inputs.input_images[index_img], out_file, self.inputs.input_masks[index_mask])
            else:
                cmd = cmd + ' -i "{}" -o "{}"" '.format(self.inputs.input_images[index_img], out_file)
        try:
            print('... cmd: {}'.format(cmd))
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath(''.join(["*", self.inputs.out_postfix, ".nii.gz"])))
        return outputs




# 
## Mask Image
# 


class MialsrtkMaskImageInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_mask = File(desc='Input mask',mandatory=True)
    out_im_postfix = traits.Str("", usedefault=True)

class MialsrtkMaskImageOutputSpec(TraitedSpec):
    out_im_file = File(desc='Masked image')

    
class MialsrtkMaskImage(BaseInterface):
    input_spec = MialsrtkMaskImageInputSpec
    output_spec = MialsrtkMaskImageOutputSpec
    
    def _run_interface(self, runtime): 
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_im_postfix, ext)))
        

        cmd = 'mialsrtkMaskImage -i "{}" -m "{}" -o "{}"'.format(self.inputs.in_file, self.inputs.in_mask, out_im_file)
        
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
        return outputs
    
    
    
class MultipleMialsrtkMaskImageInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be corrected for intensity', mandatory = True))
    out_im_postfix = traits.Str("", usedefault=True)
    stacksOrder = traits.List(mandatory = False)
    
class MultipleMialsrtkMaskImageOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleMialsrtkMaskImage(BaseInterface):
    input_spec = MultipleMialsrtkMaskImageInputSpec
    output_spec = MultipleMialsrtkMaskImageOutputSpec

    def _run_interface(self, runtime):
        
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
            index_mask = run_nb_masks.index(order)

            ax = MialsrtkMaskImage(bids_dir = self.inputs.bids_dir, in_file = self.inputs.input_images[index_img], in_mask = self.inputs.input_masks[index_mask], out_im_postfix=self.inputs.out_im_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs




# 
## Brain Extraction
# 


class BrainExtractionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='Root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_ckpt = File(desc='Network_checkpoint',mandatory=True)
    threshold = traits.Float(0.5,desc='Threshold determining cutoff probability (0.5 by default)')
    out_postfix = traits.Str("_masked.nii.gz", usedefault=True)
    #out_file = File(mandatory=True, desc= 'Output image')

class BrainExtractionOutputSpec(TraitedSpec):
    out_file = File(desc='Brain masked image',exists=True)

class BrainExtraction(BaseInterface):
    """
    Brain extraction using a pretrained U-net.
    You can read more in Salehi et al.; arXiv, 2017
    (https://arxiv.org/abs/1710.09338).
    Examples
    --------
    >>> from pymialsrtk.interfaces.preprocess import BrainExtraction
    >>> brainMask = BrainExtraction()
    >>> brainmask.inputs.base_dir = 'fixed1.nii'
    >>> brainmask.inputs.in_file = 'fixed1.nii'
    >>> brainmask.inputs.in_ckpt = 'fixed1.nii'
    >>> brainmask.inputs.threshold = 'fixed1.nii'
    >>> brainmask.inputs.out_postfix = 'fixed1.nii'
    >>> reg.run()  # doctest: +SKIP
    """

    input_spec = BrainExtractionInputSpec
    output_spec = BrainExtractionOutputSpec

    def _run_interface(self, runtime): 
        try:
            self._extract_brain(self.inputs.in_file,self.inputs.in_ckpt,
                             self.inputs.threshold,self.inputs.out_postfix)
        except Exception as e:
            print('Failed')
            print(e)    
        return runtime

    def _extract_brain(self, dataPath, modelCkpt, threshold, out_postfix):
        normalize = "local_max"
        width = 128
        height = 128
        n_channels = 1

        img_nib = nib.load(os.path.join(dataPath))
        image_data = img_nib.get_data()
        images = np.zeros((image_data.shape[2], width, height, n_channels))
        
        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(image_data[:, :, ii], dsize=(width, height), fx=width,
                                   fy=height)  # , interpolation=cv2.INTER_CUBIC)

            if normalize:
                if normalize == "local_max":
                     images[slice_counter, :, :, 0] = img_patch / np.max(img_patch)
                elif normalize == "global_max":
                     images[slice_counter, :, :, 0] = img_patch / max_val
                elif normalize ==  "mean_std":
                     images[slice_counter, :, :, 0] = (img_patch-np.mean(img_patch))/np.std(img_patch)
                else:
                     raise ValueError('Please select a valid normalization')
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

        #Tensorflow graph

        g = tf.Graph()
        with g.as_default():

            with tf.name_scope('inputs'):

               x = tf.placeholder(tf.float32, [None, width, height, n_channels])        

            conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
            pool1 = max_pool_2d(conv1, 2)

            conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
            pool2 = max_pool_2d(conv2, 2)

            conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
            pool3 = max_pool_2d(conv3, 2)

            conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
            pool4 = max_pool_2d(conv4, 2)

            conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
            conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

            up6 = upsample_2d(conv5,2)
            up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
            conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

            up7 = upsample_2d(conv6,2)
            up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
            conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

            up8 = upsample_2d(conv7,2)
            up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
            conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

            up9 = upsample_2d(conv8,2)
            up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
            conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

            pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')


        #Thresholding parameter to binarize predictions
        percentile = threshold*100

        im = np.zeros((1, width, height, n_channels))
        pred3d = []
        with tf.Session(graph=g) as sess_test:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test, modelCkpt)

            for idx in range(images.shape[0]):

                im = np.reshape(images[idx, :, :, :], [1, width, height, n_channels])

                feed_dict = {x: im}
                pred_ = sess_test.run(pred, feed_dict=feed_dict)

                theta = np.percentile(pred_,percentile)
                pred_bin = np.where(pred_>theta,1,0)
                pred3d.append(pred_bin[0, :, :, 0].astype('float64'))
            pred3d = self._post_processing(np.asarray(pred3d))
            pred3d = [cv2.resize(elem, dsize=(image_data.shape[1], image_data.shape[0]), interpolation=cv2.INTER_NEAREST) for elem in pred3d]
            pred3d = np.asarray(pred3d)
            upsampled = np.swapaxes(np.swapaxes(pred3d,1,2),0,2) #if Orient module applied, no need for this line(?)
            up_mask = nib.Nifti1Image(upsampled,img_nib.affine)
            nib.save(up_mask, dataPath.split('.')[0]+out_postfix)

    #Post-processing the binarized network output by PGD
    def _post_processing(self, pred_lbl):
        post_proc = True
        post_proc_cc = True
        post_proc_fill_holes = True

        post_proc_closing_minima = True
        post_proc_opening_maxima = True
        post_proc_extremity = False
        stackmodified = True

        crt_stack = pred_lbl.copy()
        crt_stack_pp = crt_stack.copy()

        if 1:

            distrib = []
            for iSlc in range(crt_stack.shape[0]):
                distrib.append(np.sum(crt_stack[iSlc]))

            if post_proc_cc:
                # print("post_proc_cc")
                crt_stack_cc = crt_stack.copy()
                labeled_array, num_features = snd.measurements.label(crt_stack_cc)
                unique, counts = np.unique(labeled_array, return_counts=True)

                # Try to remove false positives seen as independent connected components #2ndBrain
                for ind in range(len(unique)):
                    if 5 < counts[ind] and counts[ind] < 300:
                        wherr = np.where(labeled_array == unique[ind])
                        for ii in range(len(wherr[0])):
                            crt_stack_cc[wherr[0][ii], wherr[1][ii], wherr[2][ii]] = 0

                crt_stack_pp = crt_stack_cc.copy()

            if post_proc_fill_holes:
                # print("post_proc_fill_holes")
                crt_stack_holes = crt_stack_pp.copy()

                inv_mask = 1 - crt_stack_holes
                labeled_holes, num_holes = snd.measurements.label(inv_mask)
                unique, counts = np.unique(labeled_holes, return_counts=True)

                for lbl in unique[2:]:
                    trou = np.where(labeled_holes == lbl)
                    for ind in range(len(trou[0])):
                        inv_mask[trou[0][ind], trou[1][ind], trou[2][ind]] = 0

                crt_stack_holes = 1 - inv_mask
                crt_stack_cc = crt_stack_holes.copy()
                crt_stack_pp = crt_stack_holes.copy()

                distrib_cc = []
                for iSlc in range(crt_stack_pp.shape[0]):
                    distrib_cc.append(np.sum(crt_stack_pp[iSlc]))

            if post_proc_closing_minima or post_proc_opening_maxima:

                if 0:  # closing GLOBAL
                    crt_stack_closed_minima = crt_stack_pp.copy()
                    crt_stack_closed_minima = morphology.binary_closing(crt_stack_closed_minima)
                    crt_stack_pp = crt_stack_closed_minima.copy()

                    distrib_closed = []
                    for iSlc in range(crt_stack_closed_minima.shape[0]):
                        distrib_closed.append(np.sum(crt_stack_closed_minima[iSlc]))

                if post_proc_closing_minima:
                    # if 0:
                    crt_stack_closed_minima = crt_stack_pp.copy()

                    # for local minima
                    local_minima = argrelextrema(np.asarray(distrib_cc), np.less)[0]
                    local_maxima = argrelextrema(np.asarray(distrib_cc), np.greater)[0]

                    for iMin in range(len(local_minima)):
                        for iMax in range(len(local_maxima) - 1):
                            # print(local_maxima[iMax], "<", local_minima[iMin], "AND", local_minima[iMin], "<", local_maxima[iMax+1], "   ???")

                            # find between which maxima is the minima localized
                            if local_maxima[iMax] < local_minima[iMin] and local_minima[iMin] < local_maxima[iMax + 1]:

                                # check if diff max-min is large enough to be considered
                                if distrib_cc[local_maxima[iMax]] - distrib_cc[local_minima[iMin]] > 50 and distrib_cc[
                                    local_maxima[iMax + 1]] - distrib_cc[local_minima[iMin]] > 50:
                                    sub_stack = crt_stack_closed_minima[local_maxima[iMax] - 1:local_maxima[iMax + 1] + 1,
                                                :, :]

                                    # print("We did 3d close.")
                                    sub_stack = morphology.binary_closing(sub_stack)
                                    crt_stack_closed_minima[local_maxima[iMax] - 1:local_maxima[iMax + 1] + 1, :,
                                    :] = sub_stack

                    crt_stack_pp = crt_stack_closed_minima.copy()

                    distrib_closed = []
                    for iSlc in range(crt_stack_closed_minima.shape[0]):
                        distrib_closed.append(np.sum(crt_stack_closed_minima[iSlc]))

                if post_proc_opening_maxima:
                    crt_stack_opened_maxima = crt_stack_pp.copy()

                    local = True
                    if local:
                        local_maxima_n = argrelextrema(np.asarray(distrib_closed), np.greater)[
                            0]  # default is mode='clip'. Doesn't consider extremity as being an extrema

                        for iMax in range(len(local_maxima_n)):

                            # Check if this local maxima is a "peak"
                            if distrib[local_maxima_n[iMax]] - distrib[local_maxima_n[iMax] - 1] > 50 and distrib[
                                local_maxima_n[iMax]] - distrib[local_maxima_n[iMax] + 1] > 50:

                                if 0:
                                    print("Ceci est un pic de au moins 50.", distrib[local_maxima_n[iMax]], "en",
                                          local_maxima_n[iMax])
                                    print("                                bordé de", distrib[local_maxima_n[iMax] - 1],
                                          "en", local_maxima_n[iMax] - 1)
                                    print("                                et", distrib[local_maxima_n[iMax] + 1], "en",
                                          local_maxima_n[iMax] + 1)
                                    print("")

                                sub_stack = crt_stack_opened_maxima[local_maxima_n[iMax] - 1:local_maxima_n[iMax] + 2, :, :]
                                sub_stack = morphology.binary_opening(sub_stack)
                                crt_stack_opened_maxima[local_maxima_n[iMax] - 1:local_maxima_n[iMax] + 2, :, :] = sub_stack
                    else:
                        crt_stack_opened_maxima = morphology.binary_opening(crt_stack_opened_maxima)

                    crt_stack_pp = crt_stack_opened_maxima.copy()

                    distrib_opened = []
                    for iSlc in range(crt_stack_pp.shape[0]):
                        distrib_opened.append(np.sum(crt_stack_pp[iSlc]))

                if post_proc_extremity:

                    crt_stack_extremity = crt_stack_pp.copy()

                    # check si y a un maxima sur une extremite
                    maxima_extrema = argrelextrema(np.asarray(distrib_closed), np.greater, mode='wrap')[0]
                    # print("maxima_extrema", maxima_extrema, "     numslices",numslices, "     numslices-1",numslices-1)

                    if distrib_opened[0] - distrib_opened[1] > 40:
                        # print("First slice of ", distrib_opened, " is a maxima")
                        sub_stack = crt_stack_extremity[0:2, :, :]
                        sub_stack = morphology.binary_opening(sub_stack)
                        crt_stack_extremity[0:2, :, :] = sub_stack
                        # print("On voulait close 1st slices",  sub_stack.shape[0])

                    if pred_lbl.shape[0] - 1 in maxima_extrema:
                        # print(numslices-1, "in maxima_extrema", maxima_extrema )

                        sub_stack = crt_stack_opened_maxima[-2:, :, :]
                        sub_stack = morphology.binary_opening(sub_stack)
                        crt_stack_opened_maxima[-2:, :, :] = sub_stack

                        # print("On voulait close last slices",  sub_stack.shape[0])

                    crt_stack_pp = crt_stack_extremity.copy()

                    distrib_opened_border = []
                    for iSlc in range(crt_stack_pp.shape[0]):
                        distrib_opened_border.append(np.sum(crt_stack_pp[iSlc]))

        return crt_stack_pp

    def _list_outputs(self):

        return {'out_file': self.inputs.in_file[:-4]+self.inputs.out_postfix}
    
    
class MultipleBrainExtractionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='Root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='MRI Images', mandatory = True))
    in_ckpt = File(desc='Network_checkpoint',mandatory=True)
    threshold = traits.Float(0.5,desc='Threshold determining cutoff probability (0.5 by default)')
    out_postfix = traits.Str("_masked.nii.gz", usedefault=True)
    
class MultipleBrainExtractionOutputSpec(TraitedSpec):
    masks = OutputMultiPath(File())

class MultipleBrainExtraction(BaseInterface):
    input_spec = MultipleBrainExtractionInputSpec
    output_spec = MultipleBrainExtractionOutputSpec

    def _run_interface(self, runtime):
        #if len(self.inputs.input_images)>0:
        for input_image in self.inputs.input_images:
            ax = BrainExtraction(bids_dir = self.inputs.bids_dir, in_file = input_image, in_ckpt= self.inputs.in_ckpt,threshold = self.inputs.threshold, out_postfix=self.inputs.out_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['masks'] = glob(os.path.abspath("*.nii.gz"))
        return outputs    