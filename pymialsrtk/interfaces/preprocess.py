# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK preprocessing functions.

It includes BTK Non-local-mean denoising, slice intensity correction
slice N4 bias field correction, slice-by-slice correct bias field, intensity standardization,
histogram normalization and both manual or deep learning based automatic brain extraction.

"""

import os
import traceback
from glob import glob
import pathlib

from skimage.morphology import binary_opening, binary_closing

import numpy as np
from traits.api import *

import nibabel

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import skimage.measure
from scipy.signal import argrelextrema
import scipy.ndimage as snd
import pandas as pd
import cv2

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

from pymialsrtk.interfaces.utils import run


###############
# NLM denoising
###############

class BtkNLMDenoisingInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the BtkNLMDenoising interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    in_file = File(desc='Input image filename', mandatory=True)
    in_mask = File(desc='Input mask filename', mandatory=False)
    out_postfix = traits.Str("_nlm",
                             desc='Suffix to be added to input image filename to construst denoised output filename',
                             usedefault=True)
    weight = traits.Float(0.1,
                          desc='NLM smoothing parameter (high beta produces smoother result)',
                          usedefault=True)


class BtkNLMDenoisingOutputSpec(TraitedSpec):
    """Class used to represent outputs of the BtkNLMDenoising interface."""

    out_file = File(desc='Output denoised image file')


class BtkNLMDenoising(BaseInterface):
    """Runs the non-local mean denoising module.

    It calls the Baby toolkit implementation by Rousseau et al. [1]_ of the method proposed by Coupé et al. [2]_.

    References
    -----------
    .. [1] Rousseau et al.; Computer Methods and Programs in Biomedicine, 2013. `(link to paper) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3508300>`_
    .. [2] Coupé et al.; IEEE Transactions on Medical Imaging, 2008. `(link to paper) <https://doi.org/10.1109/tmi.2007.906087>`_

    Example
    ---------
    >>> from pymialsrtk.interfaces.preprocess import BtkNLMDenoising
    >>> nlmDenoise = BtkNLMDenoising()
    >>> nlmDenoise.inputs.bids_dir = '/my_directory'
    >>> nlmDenoise.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> nlmDenoise.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> nlmDenoise.inputs.weight = 0.2
    >>> nlmDenoise.run() # doctest: +SKIP

    """

    input_spec = BtkNLMDenoisingInputSpec
    output_spec = BtkNLMDenoisingOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = self._gen_filename('out_file')

        if self.inputs.in_mask:
            cmd = 'btkNLMDenoising -i "{}" -m "{}" -o "{}" -b {}'.format(self.inputs.in_file, self.inputs.in_mask, out_file, self.inputs.weight)
        else:
            cmd = 'btkNLMDenoising -i "{}" -o "{}" -b {}'.format(self.inputs.in_file, out_file, self.inputs.weight)

        try:
            print('... cmd: {}'.format(cmd))
            run(cmd , env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs


class MultipleBtkNLMDenoisingInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MultipleBtkNLMDenoising interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='Input image filenames to be denoised')
    input_masks = InputMultiPath(File(mandatory=False), desc='Input mask filenames')
    weight = traits.Float(0.1,
                          desc='NLM smoothing parameter (high beta produces smoother result)',
                          usedefault=True)
    out_postfix = traits.Str("_nlm",
                             desc='Suffix to be added to input image filenames to construst denoised output filenames',
                             usedefault=True)


class MultipleBtkNLMDenoisingOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleBtkNLMDenoising interface."""

    output_images = OutputMultiPath(File(), desc='Output denoised images')


class MultipleBtkNLMDenoising(BaseInterface):
    """Apply the non-local mean (NLM) denoising module on multiple inputs.

    It runs for each input image the interface :class:`pymialsrtk.interfaces.preprocess.BtkNLMDenoising`
    to the NLM denoising implementation by Rousseau et al. [1]_ of the method proposed by Coupé et al. [2]_.

    References
    ------------
    .. [1] Rousseau et al.; Computer Methods and Programs in Biomedicine, 2013. `(link to paper) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3508300>`_
    .. [2] Coupé et al.; IEEE Transactions on Medical Imaging, 2008. `(link to paper) <https://doi.org/10.1109/tmi.2007.906087>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.preprocess import MultipleBtkNLMDenoising
    >>> multiNlmDenoise = MultipleBtkNLMDenoising()
    >>> multiNlmDenoise.inputs.bids_dir = '/my_directory'
    >>> multiNlmDenoise.inputs.in_file = ['sub-01_acq-haste_run-1_T2w.nii.gz', 'sub-01_acq-haste_run-1_2w.nii.gz']
    >>> multiNlmDenoise.inputs.in_mask = ['sub-01_acq-haste_run-1_mask.nii.gz', 'sub-01_acq-haste_run-2_mask.nii.gz']
    >>> multiNlmDenoise.run() # doctest: +SKIP

    See Also
    --------
    pymialsrtk.interfaces.preprocess.BtkNLMDenoising

    """

    input_spec = MultipleBtkNLMDenoisingInputSpec
    output_spec = MultipleBtkNLMDenoisingOutputSpec

    def _run_interface(self, runtime):

        if len(self.inputs.input_masks) > 0:
            for in_image, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
                ax = BtkNLMDenoising(bids_dir=self.inputs.bids_dir,
                                     in_file=in_image,
                                     in_mask=in_mask,
                                     out_postfix=self.inputs.out_postfix,
                                     weight=self.inputs.weight)
                ax.run()
        else:
            for in_image in self.inputs.input_images:
                ax = BtkNLMDenoising(bids_dir=self.inputs.bids_dir,
                                     in_file=in_image,
                                     out_postfix=self.inputs.out_postfix,
                                     weight=self.inputs.weight)

                ax.run()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs


#############################
# Slice intensity correction
#############################

class MialsrtkCorrectSliceIntensityInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkCorrectSliceIntensity interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    in_file = File(desc='Input image filename', mandatory=True)
    in_mask = File(desc='Input mask filename', mandatory=False)
    out_postfix = traits.Str("",
                             desc='Suffix to be added to input image file to construct corrected output filename',
                             usedefault=True)


class MialsrtkCorrectSliceIntensityOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkCorrectSliceIntensity interface."""

    out_file = File(desc='Output image with corrected slice intensities')


class MialsrtkCorrectSliceIntensity(BaseInterface):
    """Runs the MIAL SRTK mean slice intensity correction module.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkCorrectSliceIntensity
    >>> sliceIntensityCorr = MialsrtkCorrectSliceIntensity()
    >>> sliceIntensityCorr.inputs.bids_dir = '/my_directory'
    >>> sliceIntensityCorr.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> sliceIntensityCorr.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> sliceIntensityCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkCorrectSliceIntensityInputSpec
    output_spec = MialsrtkCorrectSliceIntensityOutputSpec


    def _gen_filename(self, name):
        if name == 'out_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = self._gen_filename('out_file')

        cmd = 'mialsrtkCorrectSliceIntensity "{}" "{}" "{}"'.format(self.inputs.in_file, self.inputs.in_mask, out_file)
        try:
            print('... cmd: {}'.format(cmd))
            env_cpp = os.environ.copy()
            env_cpp['LD_PRELOAD'] = ""
            run(cmd, env=env_cpp, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs


class MultipleMialsrtkCorrectSliceIntensityInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MultipleMialsrtkCorrectSliceIntensity interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True),
                                  desc='Input image filenames to be corrected for slice intensity')
    input_masks = InputMultiPath(File(mandatory=False),
                                 desc='Input mask filenames')
    out_postfix = traits.Str("",
                             desc='Suffix to be added to input image filenames to construct corrected output filenames',
                             usedefault=True)


class MultipleMialsrtkCorrectSliceIntensityOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleMialsrtkCorrectSliceIntensity interface."""

    output_images = OutputMultiPath(File(), desc='Output slice intensity corrected images')


class MultipleMialsrtkCorrectSliceIntensity(BaseInterface):
    """Apply the MIAL SRTK slice intensity correction module on multiple images.

    It calls MialsrtkCorrectSliceIntensity interface with a list of images/masks.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MultipleMialsrtkCorrectSliceIntensity
    >>> multiSliceIntensityCorr = MialsrtkCorrectSliceIntensity()
    >>> multiSliceIntensityCorr.inputs.bids_dir = '/my_directory'
    >>> multiSliceIntensityCorr.inputs.in_file = ['sub-01_acq-haste_run-1_T2w.nii.gz', 'sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> multiSliceIntensityCorr.inputs.in_mask = ['sub-01_acq-haste_run-2_mask.nii.gz', 'sub-01_acq-haste_run-2_mask.nii.gz']
    >>> multiSliceIntensityCorr.run() # doctest: +SKIP

    See also
    ------------
    pymialsrtk.interfaces.preprocess.MialsrtkCorrectSliceIntensity

    """

    input_spec = MultipleMialsrtkCorrectSliceIntensityInputSpec
    output_spec = MultipleMialsrtkCorrectSliceIntensityOutputSpec

    def _run_interface(self, runtime):

        if len(self.inputs.input_masks) > 0:
            for in_image, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
                ax = MialsrtkCorrectSliceIntensity(bids_dir=self.inputs.bids_dir,
                                                   in_file=in_image,
                                                   in_mask=in_mask,
                                                   out_postfix=self.inputs.out_postfix)
                ax.run()
        else:
            for in_image in self.inputs.input_images:
                ax = MialsrtkCorrectSliceIntensity(bids_dir=self.inputs.bids_dir,
                                                   in_file=in_image,
                                                   out_postfix=self.inputs.out_postfix)
                ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs


##########################################
# Slice by slice N4 bias field correction
##########################################

class MialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkSliceBySliceN4BiasFieldCorrection interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    in_file = File(desc='Input image', mandatory=True)
    in_mask = File(desc='Input mask', mandatory=True)
    out_im_postfix = traits.Str("_bcorr",
                                desc='Suffix to be added to input image filename to construct corrected output filename',
                                usedefault=True)
    out_fld_postfix = traits.Str("_n4bias",
                                 desc='Suffix to be added to input image filename to construct output bias field filename',
                                 usedefault=True)


class MialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkSliceBySliceN4BiasFieldCorrection interface."""

    out_im_file = File(desc='Filename of corrected output image from N4 bias field (slice by slice).')
    out_fld_file = File(desc='Filename bias field extracted slice by slice from input image.')


class MialsrtkSliceBySliceN4BiasFieldCorrection(BaseInterface):
    """Runs the MIAL SRTK slice by slice N4 bias field correction module.

    This module implements the method proposed by Tustison et al. [1]_.

    References
    ------------
    .. [1] Tustison et al.; Medical Imaging, IEEE Transactions, 2010. `(link to paper) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkSliceBySliceN4BiasFieldCorrection
    >>> N4biasFieldCorr = MialsrtkSliceBySliceN4BiasFieldCorrection()
    >>> N4biasFieldCorr.inputs.bids_dir = '/my_directory'
    >>> N4biasFieldCorr.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> N4biasFieldCorr.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> N4biasFieldCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec
    output_spec = MialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec

    def _gen_filename(self, name):
        if name == 'out_im_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        elif name == 'out_fld_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_fld_postfix + ext
            if "_uni" in output:
                output.replace('_uni', '')
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = self._gen_filename('out_im_file')
        out_fld_file = self._gen_filename('out_fld_file')

        cmd = 'mialsrtkSliceBySliceN4BiasFieldCorrection "{}" "{}" "{}" "{}"'.format(self.inputs.in_file,
                                                                                     self.inputs.in_mask,
                                                                                     out_im_file, out_fld_file)
        try:
            print('... cmd: {}'.format(cmd))
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_im_file'] = self._gen_filename('out_im_file')
        outputs['out_fld_file'] = self._gen_filename('out_fld_file')
        return outputs


class MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MultipleMialsrtkSliceBySliceN4BiasFieldCorrection interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='files to be corrected for intensity')
    input_masks = InputMultiPath(File(mandatory=True), desc='mask of files to be corrected for intensity')
    out_im_postfix = traits.Str("_bcorr",
                                desc='Suffix to be added to input image filenames to construct corrected output filenames',
                                usedefault=True)
    out_fld_postfix = traits.Str("_n4bias",
                                 desc='Suffix to be added to input image filenames to construct output bias field filenames',
                                 usedefault=True)


class MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleMialsrtkSliceBySliceN4BiasFieldCorrection interface."""

    output_images = OutputMultiPath(File(), desc='Output N4 bias field corrected images')
    output_fields = OutputMultiPath(File(), desc='Output bias fields')


class MultipleMialsrtkSliceBySliceN4BiasFieldCorrection(BaseInterface):
    """Runs on multiple images the MIAL SRTK slice by slice N4 bias field correction module.

    Calls MialsrtkSliceBySliceN4BiasFieldCorrection interface that implements the method proposed by Tustison et al. [1]_ with a list of images/masks.

    References
    ------------
    .. [1] Tustison et al.; Medical Imaging, IEEE Transactions, 2010. `(link to paper) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.preprocess import MultipleMialsrtkSliceBySliceN4BiasFieldCorrection
    >>> multiN4biasFieldCorr = MialsrtkSliceBySliceN4BiasFieldCorrection()
    >>> multiN4biasFieldCorr.inputs.bids_dir = '/my_directory'
    >>> multiN4biasFieldCorr.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz', 'sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> multiN4biasFieldCorr.inputs.inputs_masks = ['sub-01_acq-haste_run-1_mask.nii.gz', 'sub-01_acq-haste_run-2_mask.nii.gz']
    >>> multiN4biasFieldCorr.run() # doctest: +SKIP

    See also
    ------------
    pymialsrtk.interfaces.preprocess.MialsrtkSliceBySliceN4BiasFieldCorrection

    """

    input_spec = MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec
    output_spec = MultipleMialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec

    def _run_interface(self, runtime):
        for in_image, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
            ax = MialsrtkSliceBySliceN4BiasFieldCorrection(bids_dir=self.inputs.bids_dir,
                                                           in_file=in_image,
                                                           in_mask=in_mask,
                                                           out_im_postfix=self.inputs.out_im_postfix,
                                                           out_fld_postfix=self.inputs.out_fld_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath(''.join(["*", self.inputs.out_im_postfix, ".nii.gz"])))
        outputs['output_fields'] = glob(os.path.abspath(''.join(["*", self.inputs.out_fld_postfix, ".nii.gz"])))
        return outputs


#####################################
# slice by slice correct bias field
#####################################

class MialsrtkSliceBySliceCorrectBiasFieldInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MialsrtkSliceBySliceCorrectBiasField interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    in_file = File(desc='Input image file', mandatory=True)
    in_mask = File(desc='Input mask file', mandatory=True)
    in_field = File(desc='Input bias field file', mandatory=True)
    out_im_postfix = traits.Str("_bcorr",
                                desc='Suffix to be added to bias field corrected `in_file`',
                                usedefault=True)


class MialsrtkSliceBySliceCorrectBiasFieldOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkSliceBySliceCorrectBiasField interface."""

    out_im_file = File(desc='Bias field corrected image')


class MialsrtkSliceBySliceCorrectBiasField(BaseInterface):
    """Runs the MIAL SRTK independant slice by slice bias field correction module.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkSliceBySliceCorrectBiasField
    >>> biasFieldCorr = MialsrtkSliceBySliceCorrectBiasField()
    >>> biasFieldCorr.inputs.bids_dir = '/my_directory'
    >>> biasFieldCorr.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> biasFieldCorr.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> biasFieldCorr.inputs.in_field = 'sub-01_acq-haste_run-1_field.nii.gz'
    >>> biasFieldCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkSliceBySliceCorrectBiasFieldInputSpec
    output_spec = MialsrtkSliceBySliceCorrectBiasFieldOutputSpec

    def _gen_filename(self, name):
        if name == 'out_im_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = self._gen_filename('out_im_file')

        cmd = 'mialsrtkSliceBySliceCorrectBiasField "{}" "{}" "{}" "{}"'.format(self.inputs.in_file, self.inputs.in_mask, self.inputs.in_field, out_im_file)
        try:
            print('... cmd: {}'.format(cmd))
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_im_file'] = self._gen_filename('out_im_file')
        return outputs


class MultipleMialsrtkSliceBySliceCorrectBiasFieldInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MultipleMialsrtkSliceBySliceCorrectBiasField interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='Files to be corrected for intensity')
    input_masks = InputMultiPath(File(mandatory=True), desc='Mask files to be corrected for intensity')
    input_fields = InputMultiPath(File(mandatory=True), desc='Bias field files to be removed', )
    out_im_postfix = traits.Str("_bcorr",
                                desc='Suffix to be added to bias field corrected input_images',
                                usedefault=True)


class MultipleMialsrtkSliceBySliceCorrectBiasFieldOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleMialsrtkSliceBySliceCorrectBiasField interface."""

    output_images = OutputMultiPath(File(), desc='Output bias field corrected images')


class MultipleMialsrtkSliceBySliceCorrectBiasField(BaseInterface):
    """Runs the MIAL SRTK slice by slice bias field correction module on multiple images.

    It calls :class:`pymialsrtk.interfaces.preprocess.MialsrtkSliceBySliceCorrectBiasField` interface
    with a list of images/masks/fields.

    Example
    ----------
    >>> from pymialsrtk.interfaces.preprocess import MultipleMialsrtkSliceBySliceN4BiasFieldCorrection
    >>> multiN4biasFieldCorr = MialsrtkSliceBySliceN4BiasFieldCorrection()
    >>> multiN4biasFieldCorr.inputs.bids_dir = '/my_directory'
    >>> multiN4biasFieldCorr.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz', 'sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> multiN4biasFieldCorr.inputs.input_masks = ['sub-01_acq-haste_run-1_mask.nii.gz', 'sub-01_acq-haste_run-2_mask.nii.gz']
    >>> multiN4biasFieldCorr.inputs.input_fields = ['sub-01_acq-haste_run-1_field.nii.gz', 'sub-01_acq-haste_run-2_field.nii.gz']
    >>> multiN4biasFieldCorr.run() # doctest: +SKIP

    See also
    ------------
    pymialsrtk.interfaces.preprocess.MialsrtkSliceBySliceCorrectBiasField

    """

    input_spec = MultipleMialsrtkSliceBySliceCorrectBiasFieldInputSpec
    output_spec = MultipleMialsrtkSliceBySliceCorrectBiasFieldOutputSpec

    def _run_interface(self, runtime):

        for in_image, in_mask, in_field in zip(self.inputs.input_images, self.inputs.input_masks, self.inputs.input_fields):
            ax = MialsrtkSliceBySliceCorrectBiasField(bids_dir=self.inputs.bids_dir,
                                                      in_file=in_image,
                                                      in_mask=in_mask,
                                                      in_field=in_field,
                                                      out_im_postfix=self.inputs.out_im_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath(''.join(["*", self.inputs.out_im_postfix, ".nii.gz"])))
        return outputs


#############################
# Intensity standardization
#############################

class MialsrtkIntensityStandardizationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkIntensityStandardization interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='Files to be corrected for intensity')
    out_postfix = traits.Str("", desc='Suffix to be added to intensity corrected input_images', usedefault=True)
    in_max = traits.Float(desc='Maximal intensity', usedefault=False)
    stacks_order = traits.List(desc='Order of images index. To ensure images are processed with their correct corresponding mask',
                               mandatory=False) # ToDo: Can be removed -> Also in pymialsrtk.pipelines.anatomical.srr.AnatomicalPipeline !!!


class MialsrtkIntensityStandardizationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkIntensityStandardization interface."""

    output_images = OutputMultiPath(File(), desc='Intensity-standardized images')


class MialsrtkIntensityStandardization(BaseInterface):
    """Runs the MIAL SRTK intensity standardization module.

    This module rescales image intensity by linear transformation

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkIntensityStandardization
    >>> intensityStandardization= MialsrtkIntensityStandardization()
    >>> intensityStandardization.inputs.bids_dir = '/my_directory'
    >>> intensityStandardization.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz','sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> intensityStandardization.run() # doctest: +SKIP

    """

    input_spec = MialsrtkIntensityStandardizationInputSpec
    output_spec = MialsrtkIntensityStandardizationOutputSpec


    def _gen_filename(self, orig, name):
        if name == 'output_images':
            _, name, ext = split_filename(orig)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        cmd = 'mialsrtkIntensityStandardization'
        for input_image in self.inputs.input_images:
            out_file = self._gen_filename(input_image, 'output_images')
            cmd = cmd + ' --input "{}" --output "{}"'.format(input_image, out_file)

        if self.inputs.in_max:
            cmd = cmd + ' --max "{}"'.format(self.inputs.in_max)

        try:
            print('... cmd: {}'.format(cmd))
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = [self._gen_filename(input_image, 'output_images') for input_image in self.inputs.input_images]
        return outputs


###########################
# Histogram normalization
###########################

class MialsrtkHistogramNormalizationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MialsrtkHistogramNormalization interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='Input image filenames to be normalized')
    input_masks = InputMultiPath(File(mandatory=False), desc='Input mask filenames')
    out_postfix = traits.Str("_histnorm",
                             desc='Suffix to be added to normalized input image filenames to construct ouptut normalized image filenames',
                             usedefault=True)


class MialsrtkHistogramNormalizationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkHistogramNormalization interface."""

    output_images = OutputMultiPath(File(), desc='Histogram-normalized images')


class MialsrtkHistogramNormalization(BaseInterface):
    """Runs the MIAL SRTK histogram normalizaton module.

    This module implements the method proposed by Nyúl et al. [1]_.

    References
    ------------
    .. [1] Nyúl et al.; Medical Imaging, IEEE Transactions, 2000. `(link to paper) <https://ieeexplore.ieee.org/document/836373>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkHistogramNormalization
    >>> histNorm = MialsrtkHistogramNormalization()
    >>> histNorm.inputs.bids_dir = '/my_directory'
    >>> histNorm.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz','sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> histNorm.inputs.input_masks = ['sub-01_acq-haste_run-1_mask.nii.gz','sub-01_acq-haste_run-2_mask.nii.gz']
    >>> histNorm.run()  # doctest: +SKIP

    """

    input_spec = MialsrtkHistogramNormalizationInputSpec
    output_spec = MialsrtkHistogramNormalizationOutputSpec

    def _gen_filename(self, orig, name):
        if name == 'output_images':
            _, name, ext = split_filename(orig)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        cmd = 'python /usr/local/bin/mialsrtkHistogramNormalization.py '

        if len(self.inputs.input_masks) > 0:
            for in_file, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
                out_file = self._gen_filename(in_file, 'output_images')
                cmd = cmd + ' -i "{}" -o "{}" -m "{}" '.format(in_file, out_file, in_mask)
        else:
            for in_file in self.inputs.input_images:
                out_file = self._gen_filename(in_file, 'output_images')
                cmd = cmd + ' -i "{}" -o "{}"" '.format(in_file, out_file)
        try:
            print('... cmd: {}'.format(cmd))
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = [self._gen_filename(in_file, 'output_images') for in_file in self.inputs.input_images]
        return outputs


##############
# Mask Image
##############

class MialsrtkMaskImageInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkMaskImage interface."""

    bids_dir = Directory(desc='BIDS root directory',mandatory=True,exists=True)
    in_file = File(desc='Input image filename to be masked',mandatory=True)
    in_mask = File(desc='Input mask filename',mandatory=True)
    out_im_postfix = traits.Str("", desc='Suffix to be added to masked in_file', usedefault=True)


class MialsrtkMaskImageOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkMaskImage interface."""

    out_im_file = File(desc='Masked image')


class MialsrtkMaskImage(BaseInterface):
    """Runs the MIAL SRTK mask image module.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkMaskImage
    >>> maskImg = MialsrtkMaskImage()
    >>> maskImg.inputs.bids_dir = '/my_directory'
    >>> maskImg.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> maskImg.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> maskImg.inputs.out_im_postfix = '_masked'
    >>> maskImg.run() # doctest: +SKIP

    """

    input_spec = MialsrtkMaskImageInputSpec
    output_spec = MialsrtkMaskImageOutputSpec

    def _gen_filename(self, name):
        if name == 'out_im_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        out_im_file = self._gen_filename('out_im_file')

        cmd = 'mialsrtkMaskImage -i "{}" -m "{}" -o "{}"'.format(self.inputs.in_file, self.inputs.in_mask, out_im_file)
        try:
            print('... cmd: {}'.format(cmd))
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_im_file'] = self._gen_filename('out_im_file')
        return outputs


class MultipleMialsrtkMaskImageInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MultipleMialsrtkMaskImage interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True),
                                  desc='Input image filenames to be corrected for intensity')
    input_masks = InputMultiPath(File(mandatory=True), desc='Input mask filenames ')
    out_im_postfix = traits.Str("", desc='Suffix to be added to masked input_images', usedefault=True)


class MultipleMialsrtkMaskImageOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleMialsrtkMaskImage interface."""

    output_images = OutputMultiPath(File(), desc='Output masked image filenames')


class MultipleMialsrtkMaskImage(BaseInterface):
    """Runs the MIAL SRTK mask image module on multiple images.

    It calls the :class:`pymialsrtk.interfaces.preprocess.MialsrtkMaskImage` interface
    with a list of images/masks.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MultipleMialsrtkMaskImage
    >>> multiMaskImg = MultipleMialsrtkMaskImage()
    >>> multiMaskImg.inputs.bids_dir = '/my_directory'
    >>> multiMaskImg.inputs.in_file = ['sub-01_acq-haste_run-1_T2w.nii.gz', 'sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> multiMaskImg.inputs.in_mask = ['sub-01_acq-haste_run-1_mask.nii.gz', 'sub-01_acq-haste_run-2_mask.nii.gz']
    >>> multiMaskImg.inputs.out_im_postfix = '_masked'
    >>> multiMaskImg.run() # doctest: +SKIP

    See also
    ------------
    pymialsrtk.interfaces.preprocess.MialsrtkMaskImage

    """

    input_spec = MultipleMialsrtkMaskImageInputSpec
    output_spec = MultipleMialsrtkMaskImageOutputSpec

    def _run_interface(self, runtime):

        for in_file, in_mask in zip(self.inputs.input_images, self.inputs.input_masks):
            ax = MialsrtkMaskImage(bids_dir=self.inputs.bids_dir,
                                   in_file=in_file,
                                   in_mask=in_mask,
                                   out_im_postfix=self.inputs.out_im_postfix)
            ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_images'] = glob(os.path.abspath("*.nii.gz"))
        return outputs


###############################
# Stacks ordering and filtering
###############################


class FilteringByRunidInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the FilteringByRunid interface."""

    input_files = InputMultiPath(File(mandatory=True),
                                 desc='Input files on which motion is computed')
    stacks_id = traits.List(desc='List of stacks id to be kept')


class FilteringByRunidOutputSpec(TraitedSpec):
    """Class used to represent outputs of the FilteringByRunid interface."""

    output_files = traits.List(desc='Filtered list of stack files')


class FilteringByRunid(BaseInterface):
    """Runs a filtering of files.

    This module filters the input files matching the specified run-ids.
    Other files are discarded.

    Examples
    --------
    >>> from pymialsrtk.interfaces.preprocess import FilteringByRunid
    >>> stacksFiltering = FilteringByRunid()
    >>> stacksFiltering.inputs.input_masks = ['sub-01_run-1_mask.nii.gz', 'sub-01_run-4_mask.nii.gz', 'sub-01_run-2_mask.nii.gz']
    >>> stacksFiltering.inputs.stacks_id = [1,2]
    >>> stacksFiltering.run() # doctest: +SKIP

    """

    input_spec = FilteringByRunidInputSpec
    output_spec = FilteringByRunidOutputSpec

    m_output_files = []

    def _run_interface(self, runtime):
        try:
            self.m_output_files = self._filter_by_runid(self.inputs.input_files, self.inputs.stacks_id)
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _filter_by_runid(self, input_files, p_stacks_id):
        output_files = []
        for f in input_files:
            f_id = int(f.split('_run-')[1].split('_')[0])
            if f_id in p_stacks_id:
                output_files.append(f)
        return output_files

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_files'] = self.m_output_files
        return outputs


class StacksOrderingInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the StacksOrdering interface."""

    input_masks = InputMultiPath(File(mandatory=True),
                                 desc='Input brain masks on which motion is computed')


class StacksOrderingOutputSpec(TraitedSpec):
    """Class used to represent outputs of the StacksOrdering interface."""

    stacks_order = traits.List(desc='Order of image `run-id` to be used for reconstruction')
    motion_tsv = File(desc='Output TSV file with results used to create `report_image`')
    report_image = File(desc='Output PNG image for report')


class StacksOrdering(BaseInterface):
    """Runs the automatic ordering of stacks.

    This module is based on the tracking of the brain mask centroid slice by slice.

    Examples
    --------
    >>> from pymialsrtk.interfaces.preprocess import StacksOrdering
    >>> stacksOrdering = StacksOrdering()
    >>> stacksOrdering.inputs.input_masks = ['sub-01_run-1_mask.nii.gz',
    >>>                                      'sub-01_run-4_mask.nii.gz',
    >>>                                      'sub-01_run-2_mask.nii.gz']
    >>> stacksOrdering.run() # doctest: +SKIP

    .. note:: In the case of discontinuous brain masks, the centroid coordinates of
        the slices excluded from the mask are set to `numpy.nan` and are not
        anymore considered in the motion index computation since `v2.0.2` release.
        Prior to this release, the centroids of these slices were set to zero
        that has shown to drastically increase the motion index with respect
        to the real motion during acquisition. However the motion in the remaining
        slices that were actually used for SR reconstruction might not correspond
        to the high value of this index.

    """

    input_spec = StacksOrderingInputSpec
    output_spec = StacksOrderingOutputSpec

    m_stack_order = []

    def _run_interface(self, runtime):
        try:
            self.m_stack_order = self._compute_stack_order()
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['stacks_order'] = self.m_stack_order
        outputs['report_image'] = os.path.abspath('motion_index_QC.png')
        outputs['motion_tsv'] = os.path.abspath('motion_index_QC.tsv')
        return outputs

    def _compute_motion_index(self, in_file):
        """Function to compute the motion index.

        The motion index is computed from the inter-slice displacement of
        the centroid of the brain mask.

        """
        central_third = True

        img = nibabel.load(in_file)
        data = img.get_fdata()

        # To compute centroid displacement as a distance
        # instead of a number of voxel
        sx, sy, sz = img.header.get_zooms()

        z = np.where(data)[2]
        data = data[..., int(min(z)):int(max(z) + 1)]

        if central_third:
            num_z = data.shape[2]
            center_z = int(num_z / 2.)

            data = data[..., int(center_z - num_z / 6.):int(center_z + num_z / 6.)]

        centroid_coord = np.zeros((data.shape[2], 2))

        for i in range(data.shape[2]):
            moments = skimage.measure.moments(data[..., i])

            try:
                centroid_coordx = moments[0, 1] / moments[0, 0]
                centroid_coordy = moments[1, 0] / moments[0, 0]
            # This happens in the case of discontinuous brain masks
            except ZeroDivisionError:
                centroid_coordx = np.nan
                centroid_coordy = np.nan

            centroid_coord[i, :] = [centroid_coordx, centroid_coordy]

        nb_of_notnans = np.count_nonzero(~np.isnan(centroid_coord))
        nb_of_nans = np.count_nonzero(np.isnan(centroid_coord))
        print(f'  Info: Number of NaNs = {nb_of_nans}')
        prop_of_nans = nb_of_nans / (nb_of_nans + nb_of_notnans)

        centroid_coord = centroid_coord[~np.isnan(centroid_coord)]
        centroid_coord = np.reshape(centroid_coord, (int(centroid_coord.shape[0] / 2), 2))

        # Zero-centering
        centroid_coord[:, 0] -= np.mean(centroid_coord[:, 0])
        centroid_coord[:, 1] -= np.mean(centroid_coord[:, 1])

        # Convert from "number of voxels" to "mm" based on the voxel size
        centroid_coord[:, 0] *= sx
        centroid_coord[:, 1] *= sy

        nb_slices = centroid_coord.shape[0]
        score = (np.var(centroid_coord[:, 0]) + np.var(centroid_coord[:, 1])) / ( nb_slices * sz)

        return score, prop_of_nans, centroid_coord[:, 0], centroid_coord[:, 1]

    def _create_report_image(self, score, prop_of_nans, centroid_coordx, centroid_coordy):
        # Output report image basename
        image_basename = 'motion_index_QC'

        print("\t>> Create report image...")
        # Visualization setup
        matplotlib.use('agg')
        sns.set_style("whitegrid")
        sns.set(font_scale=1)

        # Compute mean centroid coordinates for each image
        mean_centroid_coordx = {}
        mean_centroid_coordy = {}
        for f in self.inputs.input_masks:
            mean_centroid_coordx[f] = np.nanmean(centroid_coordx[f])
            mean_centroid_coordy[f] = np.nanmean(centroid_coordy[f])

        # Format data and create a Pandas DataFrame
        print("\t\t\t - Format data...")
        df_files = []
        df_slices = []
        df_motion_ind = []
        df_prop_of_nans = []
        df_centroid_coordx = []
        df_centroid_coordy = []
        df_centroid_displ = []

        for f in self.inputs.input_masks:
            # Extract only filename with extension from the absolute path
            path = pathlib.Path(f)
            # Extract the "run-xx" part in the filename
            fname = path.stem.split('_T2w_')[0].split('_')[1]

            for i, (coordx, coordy) in enumerate(zip(centroid_coordx[f], centroid_coordy[f])):
                df_files.append(fname)
                df_slices.append(i)
                df_motion_ind.append(score[f])
                df_prop_of_nans.append(prop_of_nans[f])
                df_centroid_coordx.append(coordx)
                df_centroid_coordy.append(coordy)
                if not np.isnan(coordx) and not np.isnan(coordy):
                    df_centroid_displ.append(
                        np.sqrt(coordx * coordx + coordy * coordy)
                    )
                else:
                    df_centroid_displ.append(np.nan)

        # Create a dataframe to facilitate handling with the results
        print("\t\t\t - Create DataFrame...")
        df = pd.DataFrame(
            {
                "Scan": df_files,
                "Slice": df_files,
                "Motion Index": df_motion_ind,
                "Proportion of NaNs (%)": df_prop_of_nans,
                "X (mm)": df_centroid_coordx,
                "Y (mm)": df_centroid_coordy,
                "Displacement Magnitude (mm)": df_centroid_displ,
            }
        )
        df = df.sort_values(by=['Motion Index', 'Scan', 'Slice'])

        # Save the results in a TSV file
        tsv_file = os.path.abspath('motion_index_QC.tsv')
        print(f"\t\t\t - Save motion results to {tsv_file}...")
        df.to_csv(tsv_file, sep='\t')

        # Make multiple figures with seaborn,
        # Saved in temporary png image and
        # combined in a final report image
        print("\t\t\t - Create figures...")

        # Show the zero-centered positions of the centroids
        sf0 = sns.jointplot(
            data=df, x="X (mm)", y="Y (mm)",
            hue="Scan",
            height=6,
        )
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + '_0.png')
        print(f'\t\t\t - Save report image 0 as {image_filename}...')
        sf0.savefig(image_filename, dpi=150)
        plt.close(sf0.fig)

        # Show the scan motion index
        sf1 = sns.catplot(
                data=df, y="Scan", x="Motion Index",
                kind="bar"
        )
        sf1.ax.set_yticklabels(
            sf1.ax.get_yticklabels(),
            rotation=0
        )
        sf1.fig.set_size_inches(6, 2)
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + '_1.png')
        print(f'\t\t\t - Save report image 1 as {image_filename}...')
        sf1.savefig(image_filename, dpi=150)
        plt.close(sf1.fig)

        # Show the displacement magnitude of the centroids
        sf2 = sns.catplot(
                data=df, y="Scan", x="Displacement Magnitude (mm)",
                kind="violin",
                inner='stick'
        )
        sf2.ax.set_yticklabels(
            sf2.ax.get_yticklabels(),
            rotation=0
        )
        sf2.fig.set_size_inches(6, 2)
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + '_2.png')
        print(f'\t\t\t - Save report image 2 as {image_filename}...')
        sf2.savefig(image_filename, dpi=150)
        plt.close(sf2.fig)

        # Show the percentage of slice with NaNs for centroids.
        # It can occur when the brain mask does not cover the slice
        sf3 = sns.catplot(
                data=df, y="Scan", x="Proportion of NaNs (%)",
                kind="bar"
        )
        sf3.ax.set_yticklabels(
            sf3.ax.get_yticklabels(),
            rotation=0
        )
        sf3.fig.set_size_inches(6, 2)
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + '_3.png')
        print(f'\t\t\t - Save report image 3 as {image_filename}...')
        sf3.savefig(image_filename, dpi=150)
        plt.close(sf3.fig)

        # Define a method to load the temporary report images
        def read_image(filename):
            """Read the PNG image with matplotlib.

            Parameters
            ----------
            filename : string
                Image filename without the absolute path

            """
            return matplotlib.image.imread(os.path.abspath(filename))

        # Create the final report image that combines the
        # four temporary report images

        fig = plt.figure(constrained_layout=True, figsize=(20, 10))

        subfigs = fig.subfigures(1, 2)

        axs = subfigs.flat[0].subplots(1, 1)
        axs.imshow(read_image(image_basename + '_0.png'))
        axs.set_axis_off()

        axs = subfigs.flat[1].subplots(3, 1)
        for i, ax in enumerate(axs):
            ax.imshow(read_image(image_basename + f'_{i+1}.png'))
            ax.set_axis_off()

        # Save the final report image
        image_filename = os.path.abspath('motion_index_QC.png')
        print(f'\t\t\t - Save final report image as {image_filename}...')
        plt.savefig(image_filename, dpi=150)

    def _compute_stack_order(self):
        """Function to compute the stacks order.

        The motion index is computed for each mask.
        Stacks are ordered according to their motion index.
        When the view plane is specified in the filenames (tag `vp`),
        stacks are ordered such that the 3 first ones are
        orthogonal / in three different orientations.
        """
        motion_ind = []

        score = {}
        prop_of_nans = {}
        centroid_coordx = {}
        centroid_coordy = {}

        for f in self.inputs.input_masks:
            score[f], prop_of_nans[f], centroid_coordx[f], centroid_coordy[f] = self._compute_motion_index(f)
            motion_ind.append(score[f])

        self._create_report_image(score, prop_of_nans, centroid_coordx, centroid_coordy)

        vp_defined = -1 not in [f.find('vp') for f in self.inputs.input_masks]
        if vp_defined:
            orientations_ = []
            for f in self.inputs.input_masks:
                orientations_.append((f.split('_vp-')[1]).split('_')[0])
            _, images_ordered, orientations_ordered = (
                list(t) for t in zip(*sorted(zip(motion_ind, self.inputs.input_masks, orientations_)))
            )
        else:
            _, images_ordered = (list(t) for t in zip(*sorted(zip(motion_ind, self.inputs.input_masks))))

        run_order = [int(f.split('run-')[1].split('_')[0]) for f in images_ordered]

        if vp_defined:
            first_ax = orientations_ordered.index('ax')
            first_sag = orientations_ordered.index('sag')
            first_cor = orientations_ordered.index('cor')
            firsts = [first_ax, first_cor, first_sag]

            run_tmp = run_order
            run_order = []
            ind_ = firsts.index(min(firsts))
            run_order.append(int(images_ordered[firsts[ind_]].split('run-')[1].split('_')[0]))

            firsts.pop(ind_)
            ind_ = firsts.index(min(firsts))
            run_order.append(int(images_ordered[firsts[ind_]].split('run-')[1].split('_')[0]))

            firsts.pop(ind_)
            ind_ = firsts.index(min(firsts))
            run_order.append(int(images_ordered[firsts[ind_]].split('run-')[1].split('_')[0]))

            others = [e for e in run_tmp if e not in run_order]
            run_order += others

        return run_order


####################
# Brain Extraction
####################


class BrainExtractionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the BrainExtraction interface."""

    bids_dir = Directory(desc='Root directory', mandatory=True, exists=True)
    in_file = File(desc='Input image', mandatory=True)
    in_ckpt_loc = File(desc='Network_checkpoint for localization', mandatory=True)
    threshold_loc = traits.Float(0.49, desc='Threshold determining cutoff probability (0.49 by default)')
    in_ckpt_seg = File(desc='Network_checkpoint for segmentation', mandatory=True)
    threshold_seg = traits.Float(0.5, desc='Threshold for cutoff probability (0.5 by default)')
    out_postfix = traits.Str("_brainMask", desc='Suffix of the automatically generated mask', usedefault=True)


class BrainExtractionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the BrainExtraction interface."""

    out_file = File(desc='Output brain mask image')


class BrainExtraction(BaseInterface):
    """Runs the automatic brain extraction module.

    This module is based on a 2D U-Net (Ronneberger et al. [1]_) using the pre-trained weights from Salehi et al. [2]_.

    References
    ------------
    .. [1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_
    .. [2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_

    Examples
    --------
    >>> from pymialsrtk.interfaces.preprocess import BrainExtraction
    >>> brainMask = BrainExtraction()
    >>> brainmask.inputs.base_dir = '/my_directory'
    >>> brainmask.inputs.in_file = 'sub-01_acq-haste_run-1_2w.nii.gz'
    >>> brainmask.inputs.in_ckpt_loc = 'my_loc_checkpoint'
    >>> brainmask.inputs.threshold_loc = 0.49
    >>> brainmask.inputs.in_ckpt_seg = 'my_seg_checkpoint'
    >>> brainmask.inputs.threshold_seg = 0.5
    >>> brainmask.inputs.out_postfix = '_brainMask.nii.gz'
    >>> brainmask.run() # doctest: +SKIP

    """

    input_spec = BrainExtractionInputSpec
    output_spec = BrainExtractionOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        try:
            self._extractBrain(self.inputs.in_file, self.inputs.in_ckpt_loc, self.inputs.threshold_loc,
                               self.inputs.in_ckpt_seg, self.inputs.threshold_seg) #, self.inputs.bids_dir, self.inputs.out_postfix)
        except Exception:
            print('Failed')
            print(traceback.format_exc())
        return runtime

    def _extractBrain(self, dataPath, modelCkptLoc, thresholdLoc, modelCkptSeg, thresholdSeg): #, bidsDir, out_postfix):
        """Generate a brain mask by passing the input image(s) through two networks.

        The first network localizes the brain by a coarse-grained segmentation while the
        second one segments it more precisely. The function saves the output mask in the
        specific module folder created in bidsDir

        Parameters
        ----------
        dataPath <string>
            Input image file (required)

        modelCkptLoc <string>
            Network_checkpoint for localization (required)

        thresholdLoc <Float>
             Threshold determining cutoff probability (default is 0.49)

        modelCkptSeg <string>
            Network_checkpoint for segmentation

        thresholdSeg <Float>
             Threshold determining cutoff probability (default is 0.5)

        bidsDir <string>
            BIDS root directory (required)

        out_postfix <string>
            Suffix of the automatically generated mask (default is '_brainMask.nii.gz')

        """
        try:
            import tflearn  # noqa: E402
            from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d  # noqa: E402
        except ImportError:
            print("tflearn not available. Can not run brain extraction")
            raise ImportError

        try:
            import tensorflow.compat.v1 as tf  # noqa: E402
        except ImportError:
            print("Tensorflow not available. Can not run brain extraction")
            raise ImportError

        # Step 1: Brain localization
        normalize = "local_max"
        width = 128
        height = 128
        border_x = 15
        border_y = 15
        n_channels = 1

        img_nib = nibabel.load(os.path.join(dataPath))
        image_data = img_nib.get_data()
        max_val = np.max(image_data)
        images = np.zeros((image_data.shape[2], width, height, n_channels))
        pred3dFinal = np.zeros((image_data.shape[2], image_data.shape[0], image_data.shape[1], n_channels))

        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(image_data[:, :, ii],
                                  dsize=(width, height),
                                  fx=width, fy=height)
            if normalize:
                if normalize == "local_max":
                    images[slice_counter, :, :, 0] = img_patch / np.max(img_patch)
                elif normalize == "global_max":
                    images[slice_counter, :, :, 0] = img_patch / max_val
                elif normalize == "mean_std":
                    images[slice_counter, :, :, 0] = (img_patch - np.mean(img_patch)) / np.std(img_patch)
                else:
                    raise ValueError('Please select a valid normalization')
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

        g = tf.Graph()
        with g.as_default():

            with tf.name_scope('inputs'):
                x = tf.placeholder(tf.float32, [None, width, height, n_channels], name='image')

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

            up6 = upsample_2d(conv5, 2)
            up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat', axis=3)
            conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

            up7 = upsample_2d(conv6, 2)
            up7 = tflearn.layers.merge_ops.merge([up7, conv3], 'concat', axis=3)
            conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

            up8 = upsample_2d(conv7, 2)
            up8 = tflearn.layers.merge_ops.merge([up8, conv2], 'concat', axis=3)
            conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

            up9 = upsample_2d(conv8, 2)
            up9 = tflearn.layers.merge_ops.merge([up9, conv1], 'concat', axis=3)
            conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

            pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')

        # Thresholding parameter to binarize predictions
        percentileLoc = thresholdLoc*100

        pred3d = []
        with tf.Session(graph=g) as sess_test_loc:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test_loc, modelCkptLoc)

            for idx in range(images.shape[0]):

                im = np.reshape(images[idx, :, :, :], [1, width, height, n_channels])

                feed_dict = {x: im}
                pred_ = sess_test_loc.run(pred, feed_dict=feed_dict)

                theta = np.percentile(pred_, percentileLoc)
                pred_bin = np.where(pred_ > theta, 1, 0)
                pred3d.append(pred_bin[0, :, :, 0].astype('float64'))

            pred3d = np.asarray(pred3d)
            heights = []
            widths = []
            coms_x = []
            coms_y = []

            # Apply PPP
            ppp = True
            if ppp:
                pred3d = self._post_processing(pred3d)

            pred3d = [cv2.resize(elem,dsize=(image_data.shape[1], image_data.shape[0]), interpolation=cv2.INTER_NEAREST) for elem in pred3d]
            pred3d = np.asarray(pred3d)
            for i in range(np.asarray(pred3d).shape[0]):
                if np.sum(pred3d[i, :, :]) != 0:
                    pred3d[i, :, :] = self._extractLargestCC(pred3d[i, :, :].astype('uint8'))
                    contours, _ = cv2.findContours(pred3d[i, :, :].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    area = cv2.minAreaRect(np.squeeze(contours))
                    heights.append(area[1][0])
                    widths.append(area[1][1])
                    bbox = cv2.boxPoints(area).astype('int')
                    coms_x.append(int((np.max(bbox[:, 1])+np.min(bbox[:, 1]))/2))
                    coms_y.append(int((np.max(bbox[:, 0])+np.min(bbox[:, 0]))/2))
            # Saving localization points
            med_x = int(np.median(coms_x))
            med_y = int(np.median(coms_y))
            half_max_x = int(np.max(heights)/2)
            half_max_y = int(np.max(widths)/2)
            x_beg = med_x-half_max_x-border_x
            x_end = med_x+half_max_x+border_x
            y_beg = med_y-half_max_y-border_y
            y_end = med_y+half_max_y+border_y

        # Step 2: Brain segmentation
        width = 96
        height = 96

        images = np.zeros((image_data.shape[2], width, height, n_channels))

        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(image_data[x_beg:x_end, y_beg:y_end, ii], dsize=(width, height))

            if normalize:
                if normalize == "local_max":
                    images[slice_counter, :, :, 0] = img_patch / np.max(img_patch)
                elif normalize == "mean_std":
                    images[slice_counter, :, :, 0] = (img_patch-np.mean(img_patch))/np.std(img_patch)
                else:
                    raise ValueError('Please select a valid normalization')
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

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

            up6 = upsample_2d(conv5, 2)
            up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
            conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

            up7 = upsample_2d(conv6, 2)
            up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
            conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

            up8 = upsample_2d(conv7, 2)
            up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
            conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

            up9 = upsample_2d(conv8, 2)
            up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
            conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

            pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')

        with tf.Session(graph=g) as sess_test_seg:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test_seg, modelCkptSeg)

            for idx in range(images.shape[0]):
                im = np.reshape(images[idx, :, :], [1, width, height, n_channels])
                feed_dict = {x: im}
                pred_ = sess_test_seg.run(pred, feed_dict=feed_dict)
                percentileSeg = thresholdSeg * 100
                theta = np.percentile(pred_, percentileSeg)
                pred_bin = np.where(pred_ > theta, 1, 0)
                # Map predictions to original indices and size
                pred_bin = cv2.resize(
                    pred_bin[0, :, :, 0],
                    dsize=(y_end-y_beg, x_end-x_beg),
                    interpolation=cv2.INTER_NEAREST)
                pred3dFinal[idx, x_beg:x_end, y_beg:y_end,0] = pred_bin.astype('float64')

            pppp = True
            if pppp:
                pred3dFinal = self._post_processing(np.asarray(pred3dFinal))
            pred3d = [
                cv2.resize(
                    elem,
                    dsize=(image_data.shape[1], image_data.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ) for elem in pred3dFinal
            ]
            pred3d = np.asarray(pred3d)
            upsampled = np.swapaxes(np.swapaxes(pred3d, 1, 2), 0, 2)  # if Orient module applied, no need for this line(?)
            up_mask = nibabel.Nifti1Image(upsampled, img_nib.affine)

            # Save output mask
            save_file = self._gen_filename('out_file')
            nibabel.save(up_mask, save_file)

    def _extractLargestCC(self, image):
        """Function returning largest connected component of an object."""

        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        # in case no segmentation
        if len(sizes) < 2:
            return image
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        largest_cc = np.zeros(output.shape)
        largest_cc[output == max_label] = 255
        return largest_cc.astype('uint8')

    def _post_processing(self, pred_lbl, verbose=False):
        """Post-processing the binarized network output by Priscille de Dumast."""

        post_proc_cc = True
        post_proc_fill_holes = True

        post_proc_closing_minima = True
        post_proc_opening_maxima = True
        post_proc_extremity = False

        crt_stack = pred_lbl.copy()
        crt_stack_pp = crt_stack.copy()

        distrib = []
        for iSlc in range(crt_stack.shape[0]):
            distrib.append(np.sum(crt_stack[iSlc]))

        if post_proc_cc:
            crt_stack_cc = crt_stack.copy()
            labeled_array, _ = snd.measurements.label(crt_stack_cc)
            unique, counts = np.unique(labeled_array, return_counts=True)

            # Try to remove false positives seen as independent connected components #2ndBrain
            for ind, _ in enumerate(unique):
                if 5 < counts[ind] < 300:
                    wherr = np.where(labeled_array == unique[ind])
                    for ii in range(len(wherr[0])):
                        crt_stack_cc[wherr[0][ii], wherr[1][ii], wherr[2][ii]] = 0

            crt_stack_pp = crt_stack_cc.copy()

        if post_proc_fill_holes:
            crt_stack_holes = crt_stack_pp.copy()

            inv_mask = 1 - crt_stack_holes
            labeled_holes, _ = snd.measurements.label(inv_mask)
            unique, counts = np.unique(labeled_holes, return_counts=True)

            for lbl in unique[2:]:
                trou = np.where(labeled_holes == lbl)
                for ind in range(len(trou[0])):
                    inv_mask[trou[0][ind], trou[1][ind], trou[2][ind]] = 0

            crt_stack_holes = 1 - inv_mask
            crt_stack_pp = crt_stack_holes.copy()

            distrib_cc = []
            for iSlc in range(crt_stack_pp.shape[0]):
                distrib_cc.append(np.sum(crt_stack_pp[iSlc]))

        if post_proc_closing_minima or post_proc_opening_maxima:

            if post_proc_closing_minima:
                crt_stack_closed_minima = crt_stack_pp.copy()

                # for local minima
                local_minima = argrelextrema(np.asarray(distrib_cc), np.less)[0]
                local_maxima = argrelextrema(np.asarray(distrib_cc), np.greater)[0]

                for iMin, _ in enumerate(local_minima):
                    for iMax in range(len(local_maxima) - 1):
                        # find between which maxima is the minima localized
                        if local_maxima[iMax] < local_minima[iMin] < local_maxima[iMax + 1]:
                            # check if diff max-min is large enough to be considered
                            if ((distrib_cc[local_maxima[iMax]] - distrib_cc[local_minima[iMin]] > 50) and
                               (distrib_cc[local_maxima[iMax + 1]] - distrib_cc[local_minima[iMin]] > 50)):
                                sub_stack = crt_stack_closed_minima[local_maxima[iMax] - 1:local_maxima[iMax + 1] + 1, :, :]
                                sub_stack = binary_closing(sub_stack)
                                crt_stack_closed_minima[local_maxima[iMax] - 1:local_maxima[iMax + 1] + 1, :, :] = sub_stack
                crt_stack_pp = crt_stack_closed_minima.copy()

                distrib_closed = []
                for iSlc in range(crt_stack_closed_minima.shape[0]):
                    distrib_closed.append(np.sum(crt_stack_closed_minima[iSlc]))

            if post_proc_opening_maxima:
                crt_stack_opened_maxima = crt_stack_pp.copy()

                local = True
                if local:
                    local_maxima_n = argrelextrema(
                        np.asarray(distrib_closed), np.greater
                    )[0]  # default is mode='clip'. Doesn't consider extremity as being an extrema

                    for iMax, _ in enumerate(local_maxima_n):
                        # Check if this local maxima is a "peak"
                        if ((distrib[local_maxima_n[iMax]] - distrib[local_maxima_n[iMax] - 1] > 50) and
                           (distrib[local_maxima_n[iMax]] - distrib[local_maxima_n[iMax] + 1] > 50)):

                            if verbose:
                                print("Ceci est un pic de au moins 50.",
                                      distrib[local_maxima_n[iMax]],
                                      "en",
                                      local_maxima_n[iMax])
                                print("                                bordé de",
                                      distrib[local_maxima_n[iMax] - 1],
                                      "en",
                                      local_maxima_n[iMax] - 1)
                                print("                                et",
                                      distrib[local_maxima_n[iMax] + 1],
                                      "en",
                                      local_maxima_n[iMax] + 1)
                                print("")

                            sub_stack = crt_stack_opened_maxima[local_maxima_n[iMax] - 1:local_maxima_n[iMax] + 2, :, :]
                            sub_stack = binary_opening(sub_stack)
                            crt_stack_opened_maxima[local_maxima_n[iMax] - 1:local_maxima_n[iMax] + 2, :, :] = sub_stack
                else:
                    crt_stack_opened_maxima = binary_opening(crt_stack_opened_maxima)
                crt_stack_pp = crt_stack_opened_maxima.copy()

                distrib_opened = []
                for iSlc in range(crt_stack_pp.shape[0]):
                    distrib_opened.append(np.sum(crt_stack_pp[iSlc]))

            if post_proc_extremity:
                crt_stack_extremity = crt_stack_pp.copy()

                # check si y a un maxima sur une extremite
                maxima_extrema = argrelextrema(np.asarray(distrib_closed),
                                               np.greater,
                                               mode='wrap')[0]

                if distrib_opened[0] - distrib_opened[1] > 40:
                    sub_stack = crt_stack_extremity[0:2, :, :]
                    sub_stack = binary_opening(sub_stack)
                    crt_stack_extremity[0:2, :, :] = sub_stack

                if pred_lbl.shape[0] - 1 in maxima_extrema:
                    sub_stack = crt_stack_opened_maxima[-2:, :, :]
                    sub_stack = binary_opening(sub_stack)
                    crt_stack_opened_maxima[-2:, :, :] = sub_stack

                crt_stack_pp = crt_stack_extremity.copy()

                distrib_opened_border = []
                for iSlc in range(crt_stack_pp.shape[0]):
                    distrib_opened_border.append(np.sum(crt_stack_pp[iSlc]))
        return crt_stack_pp

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs


class MultipleBrainExtractionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MultipleBrainExtraction interface."""

    bids_dir = Directory(desc='Root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='MRI Images')
    in_ckpt_loc = File(desc='Network_checkpoint for localization', mandatory=True)
    threshold_loc = traits.Float(0.49, desc='Threshold determining cutoff probability (0.49 by default)')
    in_ckpt_seg = File(desc='Network_checkpoint for segmentation', mandatory=True)
    threshold_seg = traits.Float(0.5, desc='Threshold determining cutoff probability (0.5 by default)')
    out_postfix = traits.Str("_brainMask", desc='Suffix of the automatically generated mask', usedefault=True)


class MultipleBrainExtractionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleBrainExtraction interface."""

    masks = OutputMultiPath(File(), desc='Output masks')


class MultipleBrainExtraction(BaseInterface):
    """Runs on multiple images the automatic brain extraction module.

    It calls on a list of images the :class:`pymialsrtk.interfaces.preprocess.BrainExtraction.BrainExtraction` module
    that implements a brain extraction algorithm based on a 2D U-Net (Ronneberger et al. [1]_) using
    the pre-trained weights from Salehi et al. [2]_.

    References
    ------------
    .. [1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_
    .. [2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_

    See also
    ------------
    pymialsrtk.interfaces.preprocess.BrainExtraction

    """

    input_spec = MultipleBrainExtractionInputSpec
    output_spec = MultipleBrainExtractionOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.input_images) > 0:
            for input_image in self.inputs.input_images:
                ax = BrainExtraction(bids_dir=self.inputs.bids_dir,
                                     in_file=input_image,
                                     in_ckpt_loc=self.inputs.in_ckpt_loc,
                                     threshold_loc=self.inputs.threshold_loc,
                                     in_ckpt_seg=self.inputs.in_ckpt_seg,
                                     threshold_seg=self.inputs.threshold_seg,
                                     out_postfix=self.inputs.out_postfix)
                ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['masks'] = glob(os.path.abspath("*.nii.gz"))
        return outputs

