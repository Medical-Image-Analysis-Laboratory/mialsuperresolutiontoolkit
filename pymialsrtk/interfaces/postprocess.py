# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK postprocessing functions.

It encompasses a High Resolution mask refinement and an N4 global bias field correction.

"""

import os

from traits.api import *

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

from pymialsrtk.interfaces.utils import run
import nibabel as nib
import numpy as np


#######################
#  Refinement HR mask
#######################

class MialsrtkRefineHRMaskByIntersectionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkRefineHRMaskByIntersection interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='Image filenames used in SR reconstruction')
    input_masks = InputMultiPath(File(mandatory=True), desc='Mask filenames')
    input_transforms = InputMultiPath(File(mandatory=True), desc='Transformation filenames')
    input_sr = File(desc='SR image filename', mandatory=True)

    input_rad_dilatation = traits.Int(1,desc='Radius of the structuring element (ball)', usedefault=True)
    in_use_staple = traits.Bool(True, desc='Use STAPLE for voting (default is True). If False, Majority voting is used instead', usedefault=True)

    out_lrmask_postfix = traits.Str("_LRmask", desc='Suffix to be added to the Low resolution input_masks',
                                    usedefault=True)
    out_srmask_postfix = traits.Str("_srMask",
                                    desc='Suffix to be added to the SR reconstruction filename to construct output SR mask filename',
                                    usedefault=True)


class MialsrtkRefineHRMaskByIntersectionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkRefineHRMaskByIntersection interface."""

    output_srmask = File(desc='Output super-resolution reconstruction refined mask')
    output_lrmasks = OutputMultiPath(File(), desc='Output low-resolution reconstruction refined masks')


class MialsrtkRefineHRMaskByIntersection(BaseInterface):
    """Runs the MIALSRTK mask refinement module.

    It uses the Simultaneous Truth And Performance Level Estimate (STAPLE) by Warfield et al. [1]_.

    References
    ------------
    .. [1] Warfield et al.; Medical Imaging, IEEE Transactions, 2004. `(link to paper) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1283110/>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.postprocess import MialsrtkRefineHRMaskByIntersection
    >>> refMask = MialsrtkRefineHRMaskByIntersection()
    >>> refMask.inputs.bids_dir = '/my_directory'
    >>> refMask.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz','sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> refMask.inputs.input_masks = ['sub-01_acq-haste_run-1_mask.nii.gz','sub-01_acq-haste_run-2_mask.nii.gz']
    >>> refMask.inputs.input_transforms = ['sub-01_acq-haste_run-1_transform.txt','sub-01_acq-haste_run-2_transform.nii.gz']
    >>> refMask.inputs.input_sr = 'sr_image.nii.gz'
    >>> refMask.run()  # doctest: +SKIP

    """

    input_spec = MialsrtkRefineHRMaskByIntersectionInputSpec
    output_spec = MialsrtkRefineHRMaskByIntersectionOutputSpec

    def _gen_filename(self, orig, name):
        if name == 'output_srmask':
            _, name, ext = split_filename(orig)
            run_id = (name.split('run-')[1]).split('_')[0]
            name = name.replace('_run-' + run_id + '_', '_')
            output = name + self.inputs.out_srmask_postfix + ext
            return os.path.abspath(output)
        elif name == 'output_lrmasks':
            _, name, ext = split_filename(orig)
            output = name + self.inputs.out_lrmask_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        cmd = ['mialsrtkRefineHRMaskByIntersection']

        cmd += ['--radius-dilation', str(self.inputs.input_rad_dilatation)]

        if self.inputs.in_use_staple:
            cmd += ['--use-staple']

        for in_file, in_mask, in_transform in zip(self.inputs.input_images, self.inputs.input_masks, self.inputs.input_transforms):

            cmd += ['-i', in_file]
            cmd += ['-m', in_mask]
            cmd += ['-t', in_transform]

            out_file = self._gen_filename(in_file, 'output_lrmasks')

            cmd += ['-O', out_file]

        out_file = self._gen_filename(self.inputs.input_images[0], 'output_srmask')

        cmd += ['-r', self.inputs.input_sr]
        cmd += ['-o', out_file]

        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_srmask'] = self._gen_filename(self.inputs.input_images[0], 'output_srmask')
        outputs['output_lrmasks'] = [self._gen_filename(in_file, 'output_lrmasks') for in_file in self.inputs.input_images]
        return outputs


############################
# N4 Bias field correction
############################

class MialsrtkN4BiasFieldCorrectionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkN4BiasFieldCorrection interface."""

    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_image = File(desc='Input image filename to be normalized', mandatory=True)
    input_mask = File(desc='Input mask filename', mandatory=False)

    out_im_postfix = traits.Str("_gbcorr", usedefault=True)
    out_fld_postfix = traits.Str("_gbcorrfield", usedefault=True)


class MialsrtkN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkN4BiasFieldCorrection interface."""

    output_image = File(desc='Output corrected image')
    output_field = File(desc='Output bias field extracted from input image')


class MialsrtkN4BiasFieldCorrection(BaseInterface):
    """
    Runs the MIAL SRTK slice by slice N4 bias field correction module.

    This tools implements the method proposed by Tustison et al. [1]_ slice by slice.

    References
    ------------
    .. [1] Tustison et al.; Medical Imaging, IEEE Transactions, 2010. `(link to paper) <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3071855>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkSliceBySliceN4BiasFieldCorrection
    >>> N4biasFieldCorr = MialsrtkSliceBySliceN4BiasFieldCorrection()
    >>> N4biasFieldCorr.inputs.bids_dir = '/my_directory'
    >>> N4biasFieldCorr.inputs.input_image = 'sub-01_acq-haste_run-1_SR.nii.gz'
    >>> N4biasFieldCorr.inputs.input_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> N4biasFieldCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkN4BiasFieldCorrectionInputSpec
    output_spec = MialsrtkN4BiasFieldCorrectionOutputSpec

    def _gen_filename(self, name):
        if name == 'output_image':
            _, name, ext = split_filename(self.inputs.input_image)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        elif name == 'output_field':
            _, name, ext = split_filename(self.inputs.input_image)
            output = name + self.inputs.out_fld_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_image))
        out_corr = self._gen_filename('output_image')
        out_fld = self._gen_filename('output_field')

        cmd = ['mialsrtkN4BiasFieldCorrection', self.inputs.input_image, self.inputs.input_mask, out_corr, out_fld]

        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, name, ext = split_filename(os.path.abspath(self.inputs.input_image))
        outputs['output_image'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                               ''.join((name, self.inputs.out_im_postfix, ext)))
        outputs['output_field'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                               ''.join((name, self.inputs.out_fld_postfix, ext)))

        return outputs


############################
# Output filenames settings
############################

class FilenamesGenerationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the FilenamesGeneration interface."""

    sub_ses = traits.Str(mandatory=True, desc='Subject and session BIDS identifier to construct output filename.')
    stacks_order = traits.List(mandatory=True, desc='List of stack run-id that specify the order of the stacks')
    sr_id = traits.Int(mandatory=True, desc='Super-Resolution id')
    use_manual_masks = traits.Bool(mandatory=True, desc='Whether masks were computed or manually performed.')


class FilenamesGenerationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the FilenamesGeneration interface."""

    substitutions = traits.List(desc='Output correspondance between old and new filenames.')


class FilenamesGeneration(BaseInterface):
    """Generates final filenames from outputs of super-resolution reconstruction.

    Example
    ----------
    >>> from pymialsrtk.interfaces.postprocess import FilenamesGeneration
    >>> filenamesGen = FilenamesGeneration()
    >>> filenamesGen.inputs.sub_ses = 'sub-01'
    >>> filenamesGen.inputs.stacks_order = [3,1,4]
    >>> filenamesGen.inputs.sr_id = 3
    >>> filenamesGen.inputs.use_manual_masks = False
    >>> filenamesGen.run() # doctest: +SKIP

    """

    input_spec = FilenamesGenerationInputSpec
    output_spec = FilenamesGenerationOutputSpec

    m_substitutions = []

    def _run_interface(self, runtime):

        for stack in self.inputs.stacks_order:
            self.m_substitutions.append(('_T2w_nlm_uni_bcorr_histnorm.nii.gz',
                                        '_id-' + str(
                                        self.inputs.sr_id) + '_desc-preprocSDI_T2w.nii.gz'))

            if not self.inputs.use_manual_masks:
                self.m_substitutions.append(('_brainExtraction0/', ''))
                self.m_substitutions.append(('_brainExtraction1/', ''))
                self.m_substitutions.append(('_brainExtraction2/', ''))
                self.m_substitutions.append(('_brainExtraction3/', ''))
                self.m_substitutions.append(('_brainExtraction4/', ''))
                self.m_substitutions.append(('_brainExtraction5/', ''))
                self.m_substitutions.append(('_brainExtraction6/', ''))
                self.m_substitutions.append(('_brainExtraction7/', ''))
                self.m_substitutions.append(('_brainExtraction8/', ''))
                self.m_substitutions.append(('_brainExtraction9/', ''))
                self.m_substitutions.append(('_T2w_brainMask.nii.gz',
                                             '_id-' + str(
                                            self.inputs.sr_id) + '_desc-brain_mask.nii.gz'))
            else:
                self.m_substitutions.append(('_T2w_mask.nii.gz',
                                             '_id-' + str(
                                            self.inputs.sr_id) + '_desc-brain_mask.nii.gz'))


            self.m_substitutions.append(('_T2w_desc-brain_', '_desc-brain_'))

            self.m_substitutions.append(('_srtkMaskImage010/', ''))
            self.m_substitutions.append(('_srtkMaskImage011/', ''))
            self.m_substitutions.append(('_srtkMaskImage012/', ''))
            self.m_substitutions.append(('_srtkMaskImage013/', ''))
            self.m_substitutions.append(('_srtkMaskImage014/', ''))
            self.m_substitutions.append(('_srtkMaskImage015/', ''))
            self.m_substitutions.append(('_srtkMaskImage016/', ''))
            self.m_substitutions.append(('_srtkMaskImage017/', ''))
            self.m_substitutions.append(('_srtkMaskImage018/', ''))
            self.m_substitutions.append(('_srtkMaskImage019/', ''))
            self.m_substitutions.append((#self.inputs.sub_ses +
                                        # '_run-' + str(stack) +
                                         '_T2w_uni_bcorr_histnorm.nii.gz',
                                        # self.inputs.sub_ses +
                                         # '_run-' + str(stack) +
                                         '_id-' + str(
                                        self.inputs.sr_id) + '_desc-preprocSR_T2w.nii.gz'))

            self.m_substitutions.append(( '_T2w_nlm_uni_bcorr_histnorm_transform_' + str(
                                        len(self.inputs.stacks_order)) + 'V.txt',
                                        '_id-' + str(
                                        self.inputs.sr_id) + '_mod-T2w_from-origin_to-SDI_mode-image_xfm.txt'))

            self.m_substitutions.append(('_run-' + str(stack) +
                                         '_T2w_uni_bcorr_histnorm_LRmask.nii.gz',
                                        '_run-' + str(stack) + '_id-' + str(
                                        self.inputs.sr_id) + '_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('SDI_' + self.inputs.sub_ses + '_' +
                                     str(len(self.inputs.stacks_order)) + 'V_rad1.nii.gz',
                                     self.inputs.sub_ses + '_rec-SDI' + '_id-' +
                                     str(self.inputs.sr_id) + '_T2w.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.inputs.sub_ses + '_' +
                                     str(len(self.inputs.stacks_order)) + 'V_rad1_gbcorr.nii.gz',
                                    self.inputs.sub_ses + '_rec-SR' + '_id-' +
                                     str(self.inputs.sr_id) + '_T2w.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.inputs.sub_ses + '_' +
                                     str(len(self.inputs.stacks_order)) + 'V_rad1.json',
                                    self.inputs.sub_ses + '_rec-SR' + '_id-' +
                                     str(self.inputs.sr_id) + '_T2w.json'))

        self.m_substitutions.append((self.inputs.sub_ses + '_T2w_uni_bcorr_histnorm_srMask.nii.gz',
                                     self.inputs.sub_ses + '_rec-SR' +
                                     '_id-' + str(self.inputs.sr_id) +
                                     '_mod-T2w_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.inputs.sub_ses +
                                     '_' + str(len(self.inputs.stacks_order)) +
                                     'V_rad1_srMask.nii.gz',
                                     self.inputs.sub_ses + '_rec-SR' +
                                     '_id-' + str(self.inputs.sr_id) +
                                     '_mod-T2w_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.inputs.sub_ses +
                                     '_' + str(len(self.inputs.stacks_order)) +
                                     'V_rad1.png',
                                     self.inputs.sub_ses + '_rec-SR' +
                                     '_id-' + str(self.inputs.sr_id) + '_T2w.png'))

        self.m_substitutions.append(('motion_index_QC.png',
                                     self.inputs.sub_ses + '_rec-SR' +
                                     '_id-' + str(self.inputs.sr_id) + '_desc-motion_stats.png'))

        self.m_substitutions.append(('motion_index_QC.tsv',
                                     self.inputs.sub_ses + '_rec-SR' +
                                     '_id-' + str(self.inputs.sr_id) + '_desc-motion_stats.tsv'))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['substitutions'] = self.m_substitutions

        return outputs


class BinarizeImageInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the BinarizeImage interface."""

    input_image = File(desc='Input image filename to be binarized', mandatory=True)


class BinarizeImageOutputSpec(TraitedSpec):
    """Class used to represent outputs of the BinarizeImage interface."""

    output_srmask = File(desc='Image mask (binarized input)')


class BinarizeImage(BaseInterface):
    """Runs the MIAL SRTK mask image module.
    Example
    =======
    >>> from pymialsrtk.interfaces.postprocess import BinarizeImage
    >>> maskImg = MialsrtkMaskImage()
    >>> maskImg.inputs.input_image = 'input_image.nii.gz'
    """

    input_spec = BinarizeImageInputSpec
    output_spec = BinarizeImageOutputSpec

    def _gen_filename(self, name):
        if name == 'output_srmask':
            _, name, ext = split_filename(self.inputs.input_image)
            output = name + '_srMask' + ext
            return os.path.abspath(output)
        return None

    def _binarize_image(self, in_image):

        image_nii = nib.load(in_image)
        image = np.asanyarray(image_nii.dataobj)

        out = nib.Nifti1Image(dataobj=1 * (image > 0), affine=image_nii.affine)
        out._header = image_nii.header

        nib.save(filename=self._gen_filename('output_srmask'), img=out)

        return

    def _run_interface(self, runtime):
        try:
            self._binarize_image(self.inputs.input_image)
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_srmask'] = self._gen_filename('output_srmask')
        return outputs
