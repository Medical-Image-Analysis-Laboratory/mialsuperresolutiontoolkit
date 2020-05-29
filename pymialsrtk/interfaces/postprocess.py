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
##  Refinement HR mask
# 

    
class MialsrtkRefineHRMaskByIntersectionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_images = InputMultiPath(File(desc='files to be SR', mandatory = True))
    input_masks = InputMultiPath(File(desc='mask of files to be SR', mandatory = True))
    input_transforms = InputMultiPath(File(desc='', mandatory = True))
    input_sr = File(mandatory=True)

    input_rad_dilatation = traits.Int(1, usedefault=True) 
    in_use_staple = traits.Bool(True, usedefault=True)

    deblurring = traits.Bool(False, usedefault=True)
    out_LRmask_postfix = traits.Str("_LRmask", usedefault=True)
    out_srmask_postfix = traits.Str("_srMask", usedefault=True)

    stacksOrder = traits.List(mandatory = False)
    
class MialsrtkRefineHRMaskByIntersectionOutputSpec(TraitedSpec):
    output_SRmask = File()
    output_LRmasks = OutputMultiPath(File())

class MialsrtkRefineHRMaskByIntersection(BaseInterface):
    input_spec = MialsrtkRefineHRMaskByIntersectionInputSpec
    output_spec = MialsrtkRefineHRMaskByIntersectionOutputSpec

    def _run_interface(self, runtime):

        cmd = ['mialsrtkRefineHRMaskByIntersection']

        cmd += ['--radius-dilation', str(self.inputs.input_rad_dilatation)]

        if self.inputs.in_use_staple:
            cmd += ['--use-staple']             

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
        for in_mask in self.inputs.input_transforms:
            cut_avt = in_mask.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_transforms.append(int(cut_apr))



        for order in self.inputs.stacksOrder:
            index_img = run_nb_images.index(order)
            index_mask = run_nb_masks.index(order)
            index_tranform = run_nb_transforms.index(order)

            cmd += ['-i', self.inputs.input_images[index_img]]
            cmd += ['-m', self.inputs.input_masks[index_mask]]
            cmd += ['-t', self.inputs.input_transforms[index_tranform]]


            _, name, ext = split_filename(self.inputs.input_images[index_img])
            out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_LRmask_postfix, ext)))
            cmd += ['-O', out_file]

        _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[0]))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_srmask_postfix, ext)))

        cmd += ['-r', self.inputs.input_sr]
        cmd += ['-o', out_file]




        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[0]))
        outputs['output_SRmask'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_srmask_postfix, ext)))
        outputs['output_LRmasks'] = glob(os.path.abspath(''.join(["*", self.inputs.out_LRmask_postfix, ext])))
        return outputs





# 
## N4 Bias field correction
# 


class MialsrtkN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    input_image = File(desc='files to be HistNorm', mandatory = True)
    input_mask = File(desc='mask of files to be HistNorm', mandatory = False)

    out_im_postfix = traits.Str("_gbcorr", usedefault=True)
    out_fld_postfix = traits.Str("_gbcorrfield", usedefault=True)

    
class MialsrtkN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    output_image = File()
    output_field = File()

class MialsrtkN4BiasFieldCorrection(BaseInterface):
    input_spec = MialsrtkN4BiasFieldCorrectionInputSpec
    output_spec = MialsrtkN4BiasFieldCorrectionOutputSpec

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_image))
        out_corr = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_im_postfix, ext)))
        out_fld = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_fld_postfix, ext)))
        
        cmd = ['mialsrtkN4BiasFieldCorrection', self.inputs.input_image, self.inputs.input_mask, out_corr, out_fld]

        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except:
            print('Failed')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_image))
        outputs['output_image'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_im_postfix, ext)))
        outputs['output_field'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir,'/fetaldata'), ''.join((name, self.inputs.out_fld_postfix, ext)))

        return outputs
