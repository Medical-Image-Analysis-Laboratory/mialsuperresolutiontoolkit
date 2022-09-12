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
import SimpleITK as sitk

import SimpleITK as sitk
import skimage.metrics
import pandas as pd
from pymialsrtk.utils import EXEC_PATH

#######################
#  Refinement HR mask
#######################

class MialsrtkRefineHRMaskByIntersectionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkRefineHRMaskByIntersection interface."""

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
    verbose = traits.Bool(desc="Enable verbosity")

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

        cmd = [f'{EXEC_PATH}mialsrtkRefineHRMaskByIntersection']

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
        if self.inputs.verbose:
            cmd += ['--verbose']
            print('... cmd: {}'.format(cmd))
        cmd = ' '.join(cmd)
        run(cmd, env={})

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

    input_image = File(desc='Input image filename to be normalized', mandatory=True)
    input_mask = File(desc='Input mask filename', mandatory=False)

    out_im_postfix = traits.Str("_gbcorr", usedefault=True)
    out_fld_postfix = traits.Str("_gbcorrfield", usedefault=True)
    verbose = traits.Bool(desc="Enable verbosity")


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
        # _, name, ext = split_filename(os.path.abspath(
        # self.inputs.input_image))
        out_corr = self._gen_filename('output_image')
        out_fld = self._gen_filename('output_field')

        cmd = [f'{EXEC_PATH}mialsrtkN4BiasFieldCorrection', self.inputs.input_image,
               self.inputs.input_mask, out_corr, out_fld]

        if self.inputs.verbose:
            cmd += [' verbose']
            print('... cmd: {}'.format(cmd))
        cmd = ' '.join(cmd)
        run(cmd, env={})
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = self._gen_filename('output_image')
        outputs['output_field'] = self._gen_filename('output_field')

        return outputs


############################
# Output filenames settings
############################

class FilenamesGenerationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the FilenamesGeneration interface."""

    stacks_order = traits.List(mandatory=True,
                               desc='List of stack run-id that specify the'
                                    ' order of the stacks')


class FilenamesGenerationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the FilenamesGeneration interface."""

    substitutions = traits.List(desc='Output correspondance between old and new filenames.')


class FilenamesGeneration(BaseInterface):
    """Generates final filenames from outputs of
    super-resolution reconstruction.

    Attributes
    ----------
    m_sub_ses :
        String containing subject-session information for output formatting
    m_sr_id :
        Super-Resolution id
    m_run_type :
        Type of run: either rec (SRRecon) or pre (Preprocessing)
    Example
    ----------
    >>> from pymialsrtk.interfaces.postprocess import FilenamesGeneration
    >>> filenamesGen = FilenamesGeneration(
        p_sub_ses = 'sub-01',
        p_sr_id = 3,
        p_run_type = "rec",
        p_use_manual_masks=False
        )
    >>> filenamesGen.inputs.sub_ses = 'sub-01'
    >>> filenamesGen.inputs.stacks_order = [3,1,4]
    >>> filenamesGen.inputs.sr_id = 3
    >>> filenamesGen.run() # doctest: +SKIP

    """

    input_spec = FilenamesGenerationInputSpec
    output_spec = FilenamesGenerationOutputSpec

    m_substitutions = []
    m_sub_ses = None
    m_sr_id = None
    m_run_type = None
    m_use_manual_masks = None

    def __init__(
            self,
            p_sub_ses,
            p_sr_id,
            p_run_type,
            p_use_manual_masks
            ):
        super().__init__()
        self.m_sub_ses = p_sub_ses
        self.m_sr_id = p_sr_id
        self.m_run_type = p_run_type
        self.m_use_manual_masks = p_use_manual_masks

    def _run_interface(self, runtime):
        run_type = self.m_run_type
        max_stacks = max(self.inputs.stacks_order)
        self.m_substitutions.append(('_T2w_nlm_uni_bcorr_histnorm.nii.gz',
                                     '_id-' + str(self.m_sr_id) +
                                     '_desc-preprocSDI_T2w.nii.gz'))

        if not self.m_use_manual_masks:
            self.m_substitutions += [(f'_brainExtraction{n}/', '')
                                     for n in range(max_stacks)]

            self.m_substitutions.append(('_T2w_brainMask.nii.gz',
                                         '_id-' + str(self.m_sr_id) +
                                         '_desc-brain_mask.nii.gz'))
        else:
            self.m_substitutions.append(('_T2w_mask.nii.gz',
                                         '_id-' + str(self.m_sr_id) +
                                         '_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('_T2w_desc-brain_', '_desc-brain_'))
        self.m_substitutions += [(f'_srtkMaskImage01{n}/', '')
                                 for n in range(max_stacks)]

        self.m_substitutions += [(f'_srtkMaskImage01_nlm{n}/', '')
                                 for n in range(max_stacks)]

        self.m_substitutions += [(f'_reduceFOV{n}/', '')
                                 for n in range(max_stacks)]

        self.m_substitutions.append(('_T2w_uni_bcorr_histnorm.nii.gz',
                                     '_id-' + str(self.m_sr_id) +
                                     '_desc-preprocSR_T2w.nii.gz'))

        self.m_substitutions.append(('_T2w_nlm_uni_bcorr_histnorm_transform_' +
                                     str(len(self.inputs.stacks_order)) +
                                     'V.txt',
                                     '_id-' + str(self.m_sr_id) +
                                     '_mod-T2w_from-origin_to-SDI_' +
                                     'mode-image_xfm.txt'))
        for stack in self.inputs.stacks_order:
            self.m_substitutions.append(('_run-' + str(stack) + '_T2w_uni_' +
                                         'bcorr_histnorm_LRmask.nii.gz',
                                         '_run-' + str(stack) + '_id-' + str(
                                         self.m_sr_id) +
                                         '_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('SDI_' + self.m_sub_ses + '_' +
                                     str(len(self.inputs.stacks_order)) +
                                     'V_rad1.nii.gz',
                                     self.m_sub_ses + f'_{run_type}-SDI' +
                                     '_id-' + str(self.m_sr_id) +
                                     '_T2w.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.m_sub_ses + '_' +
                                     str(len(self.inputs.stacks_order)) +
                                     'V_rad1_gbcorr.nii.gz',
                                    self.m_sub_ses + f'_{run_type}-SR' +
                                    '_id-' + str(self.m_sr_id) +
                                     '_T2w.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.m_sub_ses + '_' +
                                     str(len(self.inputs.stacks_order))
                                     + 'V_rad1.json',
                                    self.m_sub_ses + f'_{run_type}-SR' +
                                    '_id-' + str(self.m_sr_id) +
                                     '_T2w.json'))

        self.m_substitutions.append((self.m_sub_ses +
                                     '_T2w_uni_bcorr_histnorm_srMask.nii.gz',
                                     self.m_sub_ses + f'_{run_type}-SR' +
                                     '_id-' + str(self.m_sr_id) +
                                     '_mod-T2w_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('SDI_' + self.m_sub_ses +
                                     '_' + str(len(self.inputs.stacks_order)) +
                                     'V_rad1_srMask.nii.gz',
                                     self.m_sub_ses + '_rec-SR' +
                                     '_id-' + str(self.m_sr_id) +
                                     '_mod-T2w_desc-brain_mask.nii.gz'))

        self.m_substitutions.append(('SRTV_' + self.m_sub_ses +
                                     '_' + str(len(self.inputs.stacks_order)) +
                                     'V_rad1.png',
                                     self.m_sub_ses + f'_{run_type}-SR' +
                                     '_id-' + str(self.m_sr_id) +
                                     '_T2w.png'))

        self.m_substitutions.append((self.m_sub_ses +
                                     '_' +
                                     'HR_labelmap.nii.gz',
                                     self.m_sub_ses +
                                     '_rec-SR' +
                                     '_id-' +
                                     str(self.m_sr_id) +
                                     '_labels.nii.gz'))

        self.m_substitutions.append(('motion_index_QC.png',
                                     f'_{run_type}-SR' +
                                     '_id-' + str(self.m_sr_id) +
                                     '_desc-motion_stats.png'))

        self.m_substitutions.append(('motion_index_QC.tsv',
                                     f'_{run_type}-SR' +
                                     '_id-' + str(self.m_sr_id) +
                                     '_desc-motion_stats.tsv'))
        # Metric CSV files
        self.m_substitutions.append(
            ('SRTV_' + self.m_sub_ses + '_' +
             str(len(self.inputs.stacks_order)) +
             'V_rad1_gbcorr_trans_csv.csv',
             self.m_sub_ses + '_rec-SR' + '_id-' +
             str(self.m_sr_id) + '_desc-volume_metrics.csv'))

        self.m_substitutions.append(
            ('SRTV_' + self.m_sub_ses + '_' +
             str(len(self.inputs.stacks_order)) +
             'V_rad1_gbcorr_trans_labels_csv.csv',
             self.m_sub_ses + '_rec-SR' + '_id-' +
             str(self.m_sr_id) + '_desc-labels_metrics.csv'))

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
        self._binarize_image(self.inputs.input_image)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_srmask'] = self._gen_filename('output_srmask')
        return outputs


class ImageMetricsInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the QualityMetrics interface."""

    input_image = File(desc='Input image filename', mandatory=True)
    input_ref_image = File(
        desc='Input reference image filename',
        mandatory=True
    )
    input_ref_labelmap = File(
        desc='Input reference labelmap filename',
        mandatory=False
    )
    input_TV_parameters = traits.Dict(mandatory=True)


class ImageMetricsOutputSpec(TraitedSpec):
    """Class used to represent outputs of the QualityMetrics interface."""

    output_metrics = File(desc='Output CSV')
    output_metrics_labels = File(desc='Output per-label CSV')


class ImageMetrics(BaseInterface):
    """
    """
    input_spec = ImageMetricsInputSpec
    output_spec = ImageMetricsOutputSpec

    _image_array = None
    _reference_array = None
    _labelmap_array = None

    _dict_metrics = None
    _dict_metrics_labels = None

    def _gen_filename(self, name):
        if name == 'output_metrics':
            _, name, ext = split_filename(self.inputs.input_image)
            output = name + '_csv' + '.csv'
            return os.path.abspath(output)
        if name == 'output_metrics_labels':
            _, name, ext = split_filename(self.inputs.input_image)
            output = name + '_labels_csv' + '.csv'
            return os.path.abspath(output)
        return None

    def _reset_class_members(self):
        self._image_array = None
        self._reference_array = None
        self._labelmap_array = None

        self._dict_metrics = {}
        self._list_metrics_labels = []

    def _load_image_arrays(self):

        reader = sitk.ImageFileReader()

        reader.SetFileName(self.inputs.input_ref_image)
        self._reference_array = sitk.GetArrayFromImage(reader.Execute())

        reader.SetFileName(self.inputs.input_image)
        self._image_array = sitk.GetArrayFromImage(reader.Execute())

        if self.inputs.input_ref_labelmap is not None:
            reader.SetFileName(self.inputs.input_ref_labelmap)
            self._labelmap_array = sitk.GetArrayFromImage(reader.Execute())

    def _compute_metrics(self):

        datarange = int(
            np.amax(self._reference_array) -
            min(np.amin(self._image_array), np.amin(self._reference_array))
        )

        # print('Running PSNR computation')
        psnr = skimage.metrics.peak_signal_noise_ratio(
            self._reference_array,
            self._image_array,
            data_range=datarange
        )
        self._dict_metrics['PSNR'] = psnr

        # print('Running SSIM computation')
        ssim = skimage.metrics.structural_similarity(
            self._reference_array,
            self._image_array,
            data_range=datarange
        )
        self._dict_metrics['SSIM'] = ssim

        # print('Running nSSIM computation')
        nssim = skimage.metrics.structural_similarity(
            (self._reference_array - np.min(self._reference_array)) /
            np.ptp(self._reference_array),
            (self._image_array - np.min(self._image_array)) /
            np.ptp(self._image_array),
            data_range=1
        )
        self._dict_metrics['nSSIM'] = nssim

        # print('Running RMSE computation')
        rmse = skimage.metrics.normalized_root_mse(
            self._reference_array,
            self._image_array
        )
        self._dict_metrics['RMSE'] = rmse

    def _generate_csv(self):
        TV_params = self.inputs.input_TV_parameters

        data = []
        data.append({**TV_params, **self._dict_metrics})
        df_metrics = pd.DataFrame.from_records(data)

        df_metrics.to_csv(
            self._gen_filename('output_metrics'),
            index=False,
            header=True,
            sep=','
        )

    def _compute_metrics_labels(self):
        label_ids = list(np.unique(self._labelmap_array).astype(int))
        label_ids.remove(0)

        for label in label_ids:
            dict_label = {'Label': label}
            ref_label = np.where(
                self._labelmap_array == label,
                self._reference_array,
                0
            )
            img_label = np.where(
                self._labelmap_array == label,
                self._image_array,
                0
            )

            datarange = int(
                np.amax(ref_label) -
                min(np.amin(img_label), np.amin(ref_label))
            )

            # print('Running PSNR computation')
            psnr = skimage.metrics.peak_signal_noise_ratio(
                ref_label,
                img_label,
                data_range=datarange
            )
            dict_label['PSNR'] = psnr

            # print('Running SSIM computation')
            ssim = skimage.metrics.structural_similarity(
                ref_label,
                img_label,
                data_range=datarange
            )
            dict_label['SSIM'] = ssim

            self._list_metrics_labels.append(dict_label)
            # return

    def _generate_csv_labels(self):
        TV_params = self.inputs.input_TV_parameters

        data = []
        for it in self._list_metrics_labels:
            data.append({**TV_params, **it})
        df_metrics = pd.DataFrame.from_records(data)

        df_metrics.to_csv(
            self._gen_filename('output_metrics_labels'),
            index=False,
            header=True,
            sep=','
        )

    def _run_interface(self, runtime):
        self._reset_class_members()
        self._load_image_arrays()
        self._compute_metrics()
        self._generate_csv()
        print('Computed and saved overall metrics!')

        if self.inputs.input_ref_labelmap:
            self._compute_metrics_labels()
            self._generate_csv_labels()
            print('Computed and saved per label metrics!')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_metrics'] = \
            self._gen_filename('output_metrics')
        outputs['output_metrics_labels'] = \
            self._gen_filename('output_metrics_labels')
        return outputs


class ConcatenateImageMetricsInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of
    the ConcatenateImageMetrics interface."""

    input_metrics = InputMultiPath(File(mandatory=True), desc='')
    input_metrics_labels = InputMultiPath(File(mandatory=True), desc='')


class ConcatenateImageMetricsOutputSpec(TraitedSpec):
    """Class used to represent outputs of
    the ConcatenateImageMetrics interface."""

    output_csv = File(desc='')
    output_csv_labels = File(desc='')


class ConcatenateImageMetrics(BaseInterface):
    """ConcatenateImageMetrics

    """

    input_spec = ConcatenateImageMetricsInputSpec
    output_spec = ConcatenateImageMetricsOutputSpec

    def _gen_filename(self, name):
        if name == 'output_csv':
            return os.path.abspath(
                os.path.basename(self.inputs.input_metrics[0])
            )
        if name == 'output_csv_labels':
            return os.path.abspath(
                os.path.basename(self.inputs.input_metrics_labels[0])
            )
        return None

    def _run_interface(self, runtime):
        frames = [pd.read_csv(s, index_col=False)
                  for s in self.inputs.input_metrics]

        res = pd.concat(frames)
        res.to_csv(
            self._gen_filename('output_csv'),
            index=False,
            header=True,
            sep=','
        )

        frames_labels = [pd.read_csv(s, index_col=False)
                         for s in self.inputs.input_metrics_labels]

        res_labels = pd.concat(frames_labels)
        res_labels.to_csv(
            self._gen_filename('output_csv_labels'),
            index=False,
            header=True,
            sep=','
        )
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_csv'] = self._gen_filename('output_csv')
        outputs['output_csv_labels'] = self._gen_filename('output_csv_labels')
        return outputs


class MergeMajorityVoteInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MergeMajorityVote interface."""
    input_images = InputMultiPath(
        File(),
        desc='Inputs label-wise labelmaps to be merged',
        mandatory=True
    )


class MergeMajorityVoteOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MergeMajorityVote interface."""

    output_image = File(desc='Output label map')


class MergeMajorityVote(BaseInterface):
    """Perform majority voting to merge a list of label-wise labelmaps.
    """

    input_spec = MergeMajorityVoteInputSpec
    output_spec = MergeMajorityVoteOutputSpec

    def _gen_filename(self):
        _, name, ext = split_filename(self.inputs.input_images[0])
        output = ''.join([
            name.split('_label-')[0],
            '_labelmap',
            ext
        ])
        return os.path.abspath(output)

    def _merge_maps(self, in_images):

        in_images.sort()

        reader = sitk.ImageFileReader()

        arrays = []
        for p in in_images:
            reader.SetFileName(p)
            mask_c = reader.Execute()
            arrays.append(sitk.GetArrayFromImage(mask_c))

        maps = np.stack(arrays)
        maps = np.argmax(maps, axis=0)

        maps_sitk = sitk.GetImageFromArray(maps.astype(int))
        maps_sitk.CopyInformation(mask_c)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(self._gen_filename())

        writer.Execute(sitk.Cast(maps_sitk, sitk.sitkUInt8))

    def _run_interface(self, runtime):
        self._merge_maps(self.inputs.input_images)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = self._gen_filename()
        return outputs
