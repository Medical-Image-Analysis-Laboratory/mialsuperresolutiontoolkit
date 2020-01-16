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
        #out_file = os.path.join(self.inputs.bids_dir, ''.join((name, self.inputs.out_postfix, ext)))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))
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
        if len(self.inputs.input_masks)>0:
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
## slice by slice correct bias field 
# 


class MialsrtkSliceBySliceCorrectBiasFieldInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image',mandatory=True)
    in_mask = File(desc='Input mask',mandatory=True)
    in_field = File(desc='Input bias field',mandatory=True)
    out_im_postfix = traits.Str("_sliceFldCorrected", usedefault=True)

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
    out_im_postfix = traits.Str("_sliceFldCorrected", usedefault=True)
    
class MultipleMialsrtkSliceBySliceCorrectBiasFieldOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MultipleMialsrtkSliceBySliceCorrectBiasField(BaseInterface):
    input_spec = MultipleMialsrtkSliceBySliceCorrectBiasFieldInputSpec
    output_spec = MultipleMialsrtkSliceBySliceCorrectBiasFieldOutputSpec

    def _run_interface(self, runtime):
        for input_image, input_mask, input_field in zip(self.inputs.input_images,self.inputs.input_masks,self.inputs.input_fields):
            ax = MialsrtkSliceBySliceCorrectBiasField(bids_dir = self.inputs.bids_dir, in_file = input_image, in_mask = input_mask, in_field=input_field, out_im_postfix=self.inputs.out_im_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*_sliceFldCorrected.nii.gz"))
        return outputs




# 
## Intensity standardization 
# 
   
    
class MialsrtkIntensityStandardizationInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be corrected for intensity', mandatory = True))
    out_postfix = traits.Str("_ist", usedefault=True)
    in_max = traits.Float(usedefault=False)
    
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
        outputs['output_images'] = glob(os.path.abspath("*_ist.nii.gz"))
        return outputs



# 
## Histogram normalization 
# 


class MialsrtkHistogramNormalizationInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be HistNorm', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be HistNorm', mandatory = False))
    out_postfix = traits.Str("_HistNorm", usedefault=True)
    
class MialsrtkHistogramNormalizationOutputSpec(TraitedSpec):
    output_images = OutputMultiPath(File())

class MialsrtkHistogramNormalization(BaseInterface):
    input_spec = MialsrtkHistogramNormalizationInputSpec
    output_spec = MialsrtkHistogramNormalizationOutputSpec

    def _run_interface(self, runtime):

    	cmd = 'python /usr/local/bin/mialsrtkHistogramNormalization.py '

    	for in_file, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
    		_, name, ext = split_filename(os.path.abspath(in_file))
    		out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_postfix, ext)))
    		cmd = cmd + ' -i "{}" -o "{}" -m "{}" '.format(in_file, out_file, in_mask)

    	try:
    		print('... cmd: {}'.format(cmd))
    		run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
    	except:
    		print('Failed')

    	return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*_HistNorm.nii.gz"))
        return outputs




# 
## Image Reconstruction
# 

class MialsrtkImageReconstructionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    
    in_roi = traits.Enum(None, "all", "box", "mask", mandatory = True, default='mask', usedefault=True)
    # in_deblurring = traits.Bool(False, usedefault=True)
    # in_reg = traits.Bool(True, usedefault=True)
    # in_3d = traits.Bool(False, usedefault=True)
    
    # in_margin = traits.Float(usedefault=False)
    # in_epsilon = traits.Float(usedefault=False)
    # in_iter = traits.Int(usedefault=False)
    
    # in_combinedMasks = traits.Str(usedefault=False) ## ?? TODO
    # # in_reference = File(desc='Reference image') # , mandatory=True)

    # in_imresampled = InputMultiPath(File(desc='')) # , mandatory = True))
    # in_imroi = InputMultiPath(File(desc='')) # , mandatory = True))
    

    input_masks = InputMultiPath(File(desc='')) # , mandatory = True))
    input_images = InputMultiPath(File(desc='')) # , mandatory = True))

    out_sr_postfix = traits.Str("_sr", usedefault=True)
    out_transf_postfix = traits.Str("_transf", usedefault=True)
    stacksOrder = traits.List(mandatory=False)
    
    
class MialsrtkImageReconstructionOutputSpec(TraitedSpec):
    output_sr = File()
    output_transforms = OutputMultiPath(File(desc='SDI')) 

class MialsrtkImageReconstruction(BaseInterface):
    input_spec = MialsrtkImageReconstructionInputSpec
    output_spec = MialsrtkImageReconstructionOutputSpec

    def _run_interface(self, runtime):
        
        print("input_image", self.inputs.input_images)
        print("in roi", self.inputs.in_roi)
        print("input_masks", self.inputs.input_masks)

        params = []
        params.append(''.join(["--", self.inputs.in_roi]))
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

    #if self.inputs.in_roi == "mask":
        for in_file, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
            _, name, ext = split_filename(os.path.abspath(in_file))
            transf_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_transf_postfix, '.txt')))

            params.append("--input")
            params.append(in_file)

            params.append("--transform")
            params.append(transf_file)

            params.append("-m")
            params.append(in_mask)

        _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[0]))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, '_desc-SDI_', self.inputs.out_sr_postfix, ext)))
        params.append("--output")
        params.append(out_file)


        # if self.inputs.in_imresampled:
        #     for ir in self.inputs.in_imresampled:
        #         params.append("--ir")
        #         params.append(ir)

        # if self.inputs.in_imroi:
        #     for roi in self.inputs.in_imroi:
        #         params.append("--roi")
        #         params.append(roi)

        

        
        cmd = ["mialsrtkImageReconstruction"] 
        cmd += params
        
#         cmd = ["mialsrtkImageReconstruction", "--help"]
        
        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')
        return runtime
            
        
    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_transforms'] = glob(os.path.abspath("*.txt"))
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[0]))
        outputs['output_sr'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, '_desc-SDI_', self.inputs.out_sr_postfix, ext)))
        return outputs
