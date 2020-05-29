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
    out_im_file = File(desc='Bias field corrected image')

    
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





