# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK preprocessing functions.

It includes BTK Non-local-mean denoising, slice intensity correction
slice N4 bias field correction, slice-by-slice correct bias field, intensity standardization,
histogram normalization and both manual or deep learning based automatic brain extraction.

"""

from decimal import DivisionByZero
import os
import pathlib

from skimage.morphology import binary_opening, binary_closing

import numpy as np
from traits.api import *


# Reorientation
import SimpleITK as sitk
import nsol.principal_component_analysis as pca
from nipype.algorithms.metrics import Similarity

import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import skimage.measure
from scipy.signal import argrelextrema
import scipy.ndimage as snd
import pandas as pd
import cv2
from copy import deepcopy

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import (
    traits,
    TraitedSpec,
    File,
    InputMultiPath,
    OutputMultiPath,
    BaseInterface,
    BaseInterfaceInputSpec,
)

from pymialsrtk.interfaces.utils import run
from pymialsrtk.utils import EXEC_PATH

###############
# NLM denoising
###############


class BtkNLMDenoisingInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the BtkNLMDenoising interface."""

    in_file = File(desc="Input image filename", mandatory=True)
    in_mask = File(desc="Input mask filename", mandatory=False)
    out_postfix = traits.Str(
        "_nlm",
        desc="Suffix to be added to input image filename to construst denoised output filename",
        usedefault=True,
    )
    weight = traits.Float(
        0.1,
        desc="NLM smoothing parameter (high beta produces smoother result)",
        usedefault=True,
    )
    verbose = traits.Bool(desc="Enable verbosity")


class BtkNLMDenoisingOutputSpec(TraitedSpec):
    """Class used to represent outputs of the BtkNLMDenoising interface."""

    out_file = File(desc="Output denoised image file")


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
    >>> nlmDenoise.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> nlmDenoise.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> nlmDenoise.inputs.weight = 0.2
    >>> nlmDenoise.run() # doctest: +SKIP

    """

    input_spec = BtkNLMDenoisingInputSpec
    output_spec = BtkNLMDenoisingOutputSpec

    def _gen_filename(self, name):
        if name == "out_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = self._gen_filename("out_file")
        if self.inputs.in_mask:
            cmd = (
                f'{EXEC_PATH}btkNLMDenoising -i "{self.inputs.in_file}" '
                f'-m "{self.inputs.in_mask}" -o "{out_file}" '
                f"-b {self.inputs.weight}"
            )
        else:
            cmd = (
                f'{EXEC_PATH}btkNLMDenoising -i "{self.inputs.in_file}" '
                f'-o "{out_file}" -b {self.inputs.weight}'
            )
        if self.inputs.verbose:
            cmd += " --verbose"
            print("... cmd: {}".format(cmd))
        run(cmd, env={})
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = self._gen_filename("out_file")
        return outputs


#############################
# Slice intensity correction
#############################


class MialsrtkCorrectSliceIntensityInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkCorrectSliceIntensity interface."""

    in_file = File(desc="Input image filename", mandatory=True)
    in_mask = File(desc="Input mask filename", mandatory=False)
    out_postfix = traits.Str(
        "",
        desc="Suffix to be added to input image file to construct corrected output filename",
        usedefault=True,
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkCorrectSliceIntensityOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkCorrectSliceIntensity interface."""

    out_file = File(desc="Output image with corrected slice intensities")


class MialsrtkCorrectSliceIntensity(BaseInterface):
    """Runs the MIAL SRTK mean slice intensity correction module.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkCorrectSliceIntensity
    >>> sliceIntensityCorr = MialsrtkCorrectSliceIntensity()
    >>> sliceIntensityCorr.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> sliceIntensityCorr.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> sliceIntensityCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkCorrectSliceIntensityInputSpec
    output_spec = MialsrtkCorrectSliceIntensityOutputSpec

    def _gen_filename(self, name):
        if name == "out_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_file = self._gen_filename("out_file")

        cmd = (
            f"{EXEC_PATH}mialsrtkCorrectSliceIntensity "
            f'"{self.inputs.in_file}" "{self.inputs.in_mask}" "{out_file}"'
        )
        if self.inputs.verbose:
            cmd += " verbose"
            print("... cmd: {}".format(cmd))
        env_cpp = os.environ.copy()
        env_cpp["LD_PRELOAD"] = ""
        run(cmd, env=env_cpp)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = self._gen_filename("out_file")
        return outputs


##########################################
# Slice by slice N4 bias field correction
##########################################


class MialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec(
    BaseInterfaceInputSpec
):
    """Class used to represent inputs of the MialsrtkSliceBySliceN4BiasFieldCorrection interface."""

    in_file = File(desc="Input image", mandatory=True)
    in_mask = File(desc="Input mask", mandatory=True)
    out_im_postfix = traits.Str(
        "_bcorr",
        desc="Suffix to be added to input image filename to construct corrected output filename",
        usedefault=True,
    )
    out_fld_postfix = traits.Str(
        "_n4bias",
        desc="Suffix to be added to input image filename to construct output bias field filename",
        usedefault=True,
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkSliceBySliceN4BiasFieldCorrection interface."""

    out_im_file = File(
        desc="Filename of corrected output image from N4 bias field (slice by slice)."
    )
    out_fld_file = File(
        desc="Filename bias field extracted slice by slice from input image."
    )


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
    >>> N4biasFieldCorr.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> N4biasFieldCorr.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> N4biasFieldCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkSliceBySliceN4BiasFieldCorrectionInputSpec
    output_spec = MialsrtkSliceBySliceN4BiasFieldCorrectionOutputSpec

    def _gen_filename(self, name):
        if name == "out_im_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        elif name == "out_fld_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_fld_postfix + ext
            if "_uni" in output:
                output.replace("_uni", "")
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = self._gen_filename("out_im_file")
        out_fld_file = self._gen_filename("out_fld_file")

        cmd = (
            f"{EXEC_PATH}mialsrtkSliceBySliceN4BiasFieldCorrection "
            f'"{self.inputs.in_file}" "{self.inputs.in_mask}" '
            f'"{out_im_file}" "{out_fld_file}"'
        )
        if self.inputs.verbose:
            cmd += " verbose"
            print("... cmd: {}".format(cmd))
        run(cmd, env={})

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_im_file"] = self._gen_filename("out_im_file")
        outputs["out_fld_file"] = self._gen_filename("out_fld_file")
        return outputs


#####################################
# slice by slice correct bias field
#####################################


class MialsrtkSliceBySliceCorrectBiasFieldInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MialsrtkSliceBySliceCorrectBiasField interface."""

    in_file = File(desc="Input image file", mandatory=True)
    in_mask = File(desc="Input mask file", mandatory=True)
    in_field = File(desc="Input bias field file", mandatory=True)
    out_im_postfix = traits.Str(
        "_bcorr",
        desc="Suffix to be added to bias field corrected `in_file`",
        usedefault=True,
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkSliceBySliceCorrectBiasFieldOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkSliceBySliceCorrectBiasField interface."""

    out_im_file = File(desc="Bias field corrected image")


class MialsrtkSliceBySliceCorrectBiasField(BaseInterface):
    """Runs the MIAL SRTK independant slice by slice bias field correction module.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkSliceBySliceCorrectBiasField
    >>> biasFieldCorr = MialsrtkSliceBySliceCorrectBiasField()
    >>> biasFieldCorr.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> biasFieldCorr.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> biasFieldCorr.inputs.in_field = 'sub-01_acq-haste_run-1_field.nii.gz'
    >>> biasFieldCorr.run() # doctest: +SKIP

    """

    input_spec = MialsrtkSliceBySliceCorrectBiasFieldInputSpec
    output_spec = MialsrtkSliceBySliceCorrectBiasFieldOutputSpec

    def _gen_filename(self, name):
        if name == "out_im_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        _, name, ext = split_filename(os.path.abspath(self.inputs.in_file))
        out_im_file = self._gen_filename("out_im_file")

        cmd = (
            f"{EXEC_PATH}mialsrtkSliceBySliceCorrectBiasField "
            f'"{self.inputs.in_file}" "{self.inputs.in_mask}" '
            f'"{self.inputs.in_field}" "{out_im_file}"'
        )
        if self.inputs.verbose:
            cmd += " verbose"
            print("... cmd: {}".format(cmd))
        run(cmd, env={})
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_im_file"] = self._gen_filename("out_im_file")
        return outputs


#############################
# Intensity standardization
#############################


class MialsrtkIntensityStandardizationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkIntensityStandardization interface."""

    input_images = InputMultiPath(
        File(mandatory=True), desc="Files to be corrected for intensity"
    )
    out_postfix = traits.Str(
        "",
        desc="Suffix to be added to intensity corrected input_images",
        usedefault=True,
    )
    in_max = traits.Float(desc="Maximal intensity", usedefault=False)
    stacks_order = traits.List(
        desc="Order of images index. To ensure images are processed with their correct corresponding mask",
        mandatory=False,
    )  # ToDo: Can be removed -> Also in pymialsrtk.pipelines.anatomical.srr.AnatomicalPipeline !!!
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkIntensityStandardizationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkIntensityStandardization interface."""

    output_images = OutputMultiPath(
        File(), desc="Intensity-standardized images"
    )


class MialsrtkIntensityStandardization(BaseInterface):
    """Runs the MIAL SRTK intensity standardization module.

    This module rescales image intensity by linear transformation

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkIntensityStandardization
    >>> intensityStandardization= MialsrtkIntensityStandardization()
    >>> intensityStandardization.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz','sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> intensityStandardization.run() # doctest: +SKIP

    """

    input_spec = MialsrtkIntensityStandardizationInputSpec
    output_spec = MialsrtkIntensityStandardizationOutputSpec

    def _gen_filename(self, orig, name):
        if name == "output_images":
            _, name, ext = split_filename(orig)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        cmd = f"{EXEC_PATH}mialsrtkIntensityStandardization"
        for input_image in self.inputs.input_images:
            out_file = self._gen_filename(input_image, "output_images")
            cmd = cmd + ' --input "{}" --output "{}"'.format(
                input_image, out_file
            )

        if self.inputs.in_max:
            cmd = cmd + ' --max "{}"'.format(self.inputs.in_max)

        if self.inputs.verbose:
            cmd = cmd + " --verbose"
            print("... cmd: {}".format(cmd))
        run(cmd, env={})
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_images"] = [
            self._gen_filename(input_image, "output_images")
            for input_image in self.inputs.input_images
        ]
        return outputs


###########################
# Histogram normalization
###########################


class MialsrtkHistogramNormalizationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MialsrtkHistogramNormalization interface."""

    input_images = InputMultiPath(
        File(mandatory=True), desc="Input image filenames to be normalized"
    )
    input_masks = InputMultiPath(
        File(mandatory=False), desc="Input mask filenames"
    )
    out_postfix = traits.Str(
        "_histnorm",
        desc="Suffix to be added to normalized input image filenames to construct ouptut normalized image filenames",
        usedefault=True,
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkHistogramNormalizationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkHistogramNormalization interface."""

    output_images = OutputMultiPath(File(), desc="Histogram-normalized images")


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
    >>> histNorm.inputs.input_images = ['sub-01_acq-haste_run-1_T2w.nii.gz','sub-01_acq-haste_run-2_T2w.nii.gz']
    >>> histNorm.inputs.input_masks = ['sub-01_acq-haste_run-1_mask.nii.gz','sub-01_acq-haste_run-2_mask.nii.gz']
    >>> histNorm.run()  # doctest: +SKIP

    """

    input_spec = MialsrtkHistogramNormalizationInputSpec
    output_spec = MialsrtkHistogramNormalizationOutputSpec

    def _gen_filename(self, orig, name):
        if name == "output_images":
            _, name, ext = split_filename(orig)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime, verbose=False):

        cmd = "python /usr/local/bin/mialsrtkHistogramNormalization.py "

        if len(self.inputs.input_masks) > 0:
            for in_file, in_mask in zip(
                self.inputs.input_images, self.inputs.input_masks
            ):
                out_file = self._gen_filename(in_file, "output_images")
                cmd = cmd + ' -i "{}" -o "{}" -m "{}" '.format(
                    in_file, out_file, in_mask
                )
        else:
            for in_file in self.inputs.input_images:
                out_file = self._gen_filename(in_file, "output_images")
                cmd = cmd + ' -i "{}" -o "{}" '.format(in_file, out_file)
        if self.inputs.verbose:
            cmd += " -v"
            print("... cmd: {}".format(cmd))
        run(cmd, env={})

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_images"] = [
            self._gen_filename(in_file, "output_images")
            for in_file in self.inputs.input_images
        ]
        return outputs


##############
# Mask Image
##############


class MialsrtkMaskImageInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkMaskImage interface."""

    in_file = File(desc="Input image filename to be masked", mandatory=True)
    in_mask = File(desc="Input mask filename", mandatory=True)
    out_im_postfix = traits.Str(
        "", desc="Suffix to be added to masked in_file", usedefault=True
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkMaskImageOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkMaskImage interface."""

    out_im_file = File(desc="Masked image")


class MialsrtkMaskImage(BaseInterface):
    """Runs the MIAL SRTK mask image module.

    Example
    =======
    >>> from pymialsrtk.interfaces.preprocess import MialsrtkMaskImage
    >>> maskImg = MialsrtkMaskImage()
    >>> maskImg.inputs.in_file = 'sub-01_acq-haste_run-1_T2w.nii.gz'
    >>> maskImg.inputs.in_mask = 'sub-01_acq-haste_run-1_mask.nii.gz'
    >>> maskImg.inputs.out_im_postfix = '_masked'
    >>> maskImg.run() # doctest: +SKIP

    """

    input_spec = MialsrtkMaskImageInputSpec
    output_spec = MialsrtkMaskImageOutputSpec

    def _gen_filename(self, name):
        if name == "out_im_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_im_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        out_im_file = self._gen_filename("out_im_file")

        cmd = (
            f'{EXEC_PATH}mialsrtkMaskImage -i "{self.inputs.in_file}" '
            f'-m "{self.inputs.in_mask}" -o "{out_im_file}"'
        )
        if self.inputs.verbose:
            cmd += " --verbose"
            print("... cmd: {}".format(cmd))
        run(cmd, env={})

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_im_file"] = self._gen_filename("out_im_file")
        return outputs


###############################
# Stacks ordering and filtering
###############################


class CheckAndFilterInputStacksInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the FilterInputStacks interface."""

    input_images = InputMultiPath(File(mandatory=True), desc="Input images")
    input_masks = InputMultiPath(File(None), desc="Input masks")
    input_labels = InputMultiPath(File(None), desc="Input label maps")
    stacks_id = traits.List(desc="List of stacks id to be kept")


class CheckAndFilterInputStacksOutputSpec(TraitedSpec):
    """Class used to represent outputs of the FilterInputStacks interface."""

    output_stacks = traits.List(desc="Filtered list of stack files")
    output_images = traits.List(
        traits.Str, desc="Filtered list of image files"
    )
    output_masks = traits.List(desc="Filtered list of mask files")
    output_labels = traits.List(desc="Filtered list of label files")


class CheckAndFilterInputStacks(BaseInterface):
    """Runs a filtering and a check on the input files.

    This module filters the input files matching the specified run-ids.
    Other files are discarded.

    Examples
    --------
    >>> from pymialsrtk.interfaces.preprocess import CheckAndFilterInputStacks
    >>> stacksFiltering = CheckAndFilterInputStacks()
    >>> stacksFiltering.inputs.input_masks = ['sub-01_run-1_mask.nii.gz', 'sub-01_run-4_mask.nii.gz', 'sub-01_run-2_mask.nii.gz']
    >>> stacksFiltering.inputs.stacks_id = [1,2]
    >>> stacksFiltering.run() # doctest: +SKIP

    """

    input_spec = CheckAndFilterInputStacksInputSpec
    output_spec = CheckAndFilterInputStacksOutputSpec

    m_output_stacks = []
    m_output_images = []
    m_output_masks = []
    m_output_labels = []

    def _run_interface(self, runtime):
        self.m_output_stacks, out_files = self._filter_by_runid(
            self.inputs.input_images,
            self.inputs.input_masks,
            self.inputs.input_labels,
            self.inputs.stacks_id,
        )
        self.m_output_images = out_files.pop(0)
        if self.inputs.input_masks:
            self.m_output_masks = out_files.pop(0)
        if self.inputs.input_labels:
            self.m_output_labels = out_files.pop(0)

        return runtime

    def _filter_by_runid(
        self, input_images, input_masks, input_labels, p_stacks_id
    ):

        input_checks = [input_images]
        if input_masks:
            input_checks.append(input_masks)
        if input_labels:
            input_checks.append(input_labels)

        if p_stacks_id:
            assert len(p_stacks_id) > 1, (
                f"Only a single stack (# {p_stacks_id[0]}) "
                "was given. MialSRTK needs at least two stacks to run."
            )
        else:
            # If stacks aren't given, take as stack the runs found in the images.
            # 1. Check that there is at least two scans in the input folder.
            assert len(input_images) > 1, (
                f"Only a single input file ({input_images[0]}) "
                "was found. MialSRTK needs at least two stacks to run.\n"
                "It is however recommended to use at least *three* orthogonal stacks."
            )
            p_stacks_id = [
                int(f.split("run-")[1].split("_")[0]) for f in input_images
            ]

        # Check consistency between files, i.e. that for a given p_stacks_id
        # the file exists for each inputs.
        output_files = []
        for input_files in input_checks:
            stacks = deepcopy(p_stacks_id)
            output_list = []
            for f in input_files:
                f_id = int(f.split("_run-")[1].split("_")[0])
                if f_id in p_stacks_id:
                    output_list.append(f)
                    stacks.remove(f_id)
            output_files.append(output_list)
            if len(stacks) > 0:
                raise RuntimeError(
                    f"Stacks with id {stacks} not found in {os.path.dirname(f)}."
                )
        return p_stacks_id, output_files

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_stacks"] = self.m_output_stacks
        outputs["output_images"] = self.m_output_images
        outputs["output_masks"] = self.m_output_masks
        outputs["output_labels"] = self.m_output_labels
        return outputs


class StacksOrderingInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the StacksOrdering interface."""

    input_masks = InputMultiPath(
        File(mandatory=True),
        desc="Input brain masks on which motion is computed",
    )
    sub_ses = traits.Str(
        desc=("Subject and session BIDS identifier"), mandatory=True
    )
    verbose = traits.Bool(desc="Enable verbosity")


class StacksOrderingOutputSpec(TraitedSpec):
    """Class used to represent outputs of the StacksOrdering interface."""

    stacks_order = traits.List(
        desc="Order of image `run-id` to be used for reconstruction"
    )
    motion_tsv = File(
        desc="Output TSV file with results used to create `report_image`"
    )
    report_image = File(desc="Output PNG image for report")


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

    def _gen_filename(self, name):
        if name == "report_image":
            output = self.inputs.sub_ses + "_motion_index_QC.png"
            return os.path.abspath(output)
        elif name == "motion_tsv":
            output = self.inputs.sub_ses + "_motion_index_QC.tsv"
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        self.m_stack_order = self._compute_stack_order()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["stacks_order"] = self.m_stack_order
        outputs["report_image"] = self._gen_filename("report_image")
        outputs["motion_tsv"] = self._gen_filename("motion_tsv")
        return outputs

    def _compute_motion_index(self, in_file):
        """Function to compute the motion index.

        The motion index is computed from the inter-slice displacement of
        the centroid of the brain mask.

        """
        central_third = True

        img = nib.load(in_file)
        data = img.get_fdata()

        # To compute centroid displacement as a distance
        # instead of a number of voxel
        sx, sy, sz = img.header.get_zooms()

        z = np.where(data)[2]
        data = data[..., int(min(z)) : int(max(z) + 1)]

        if central_third:
            num_z = data.shape[2]
            center_z = int(num_z / 2.0)

            data = data[
                ..., int(center_z - num_z / 6.0) : int(center_z + num_z / 6.0)
            ]

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
        if nb_of_nans > 0:
            print(f"  Info: File {in_file} - Number of NaNs = {nb_of_nans}")

        if nb_of_nans + nb_of_notnans == 0:
            import re

            run_id = re.findall(r"run-(\d+)_", in_file)[-1]
            raise DivisionByZero(
                f"The mask of run-{run_id} is empty on the range "
                "considered. The stack should be excluded."
            )

        prop_of_nans = nb_of_nans / (nb_of_nans + nb_of_notnans)

        centroid_coord = centroid_coord[~np.isnan(centroid_coord)]
        centroid_coord = np.reshape(
            centroid_coord, (int(centroid_coord.shape[0] / 2), 2)
        )

        # Zero-centering
        centroid_coord[:, 0] -= np.mean(centroid_coord[:, 0])
        centroid_coord[:, 1] -= np.mean(centroid_coord[:, 1])

        # Convert from "number of voxels" to "mm" based on the voxel size
        centroid_coord[:, 0] *= sx
        centroid_coord[:, 1] *= sy

        nb_slices = centroid_coord.shape[0]
        score = (
            np.var(centroid_coord[:, 0]) + np.var(centroid_coord[:, 1])
        ) / (nb_slices * sz)

        return score, prop_of_nans, centroid_coord[:, 0], centroid_coord[:, 1]

    def _create_report_image(
        self, score, prop_of_nans, centroid_coordx, centroid_coordy
    ):
        # Output report image basename
        image_basename = "motion_index_QC"
        if self.inputs.verbose:
            print("\t>> Create report image...")
        # Visualization setup
        matplotlib.use("agg")
        sns.set_style("whitegrid")
        sns.set(font_scale=1)

        # Compute mean centroid coordinates for each image
        mean_centroid_coordx = {}
        mean_centroid_coordy = {}
        for f in self.inputs.input_masks:
            mean_centroid_coordx[f] = np.nanmean(centroid_coordx[f])
            mean_centroid_coordy[f] = np.nanmean(centroid_coordy[f])

        # Format data and create a Pandas DataFrame
        if self.inputs.verbose:
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
            fname = path.stem.split("_T2w_")[0].split("_")[1]

            for i, (coordx, coordy) in enumerate(
                zip(centroid_coordx[f], centroid_coordy[f])
            ):
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
        if self.inputs.verbose:
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
        df = df.sort_values(by=["Motion Index", "Scan", "Slice"])

        # Save the results in a TSV file
        tsv_file = self._gen_filename("motion_tsv")
        if self.inputs.verbose:
            print(f"\t\t\t - Save motion results to {tsv_file}...")
        df.to_csv(tsv_file, sep="\t")

        # Make multiple figures with seaborn,
        # Saved in temporary png image and
        # combined in a final report image
        if self.inputs.verbose:
            print("\t\t\t - Create figures...")

        # Show the zero-centered positions of the centroids
        sf0 = sns.jointplot(
            data=df,
            x="X (mm)",
            y="Y (mm)",
            hue="Scan",
            height=6,
        )
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + "_0.png")
        if self.inputs.verbose:
            print(f"\t\t\t - Save report image 0 as {image_filename}...")
        sf0.savefig(image_filename, dpi=150)
        plt.close(sf0.fig)

        # Show the scan motion index
        sf1 = sns.catplot(data=df, y="Scan", x="Motion Index", kind="bar")
        sf1.ax.set_yticklabels(sf1.ax.get_yticklabels(), rotation=0)
        sf1.fig.set_size_inches(6, 2)
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + "_1.png")
        if self.inputs.verbose:
            print(f"\t\t\t - Save report image 1 as {image_filename}...")
        sf1.savefig(image_filename, dpi=150)
        plt.close(sf1.fig)

        # Show the displacement magnitude of the centroids
        sf2 = sns.catplot(
            data=df,
            y="Scan",
            x="Displacement Magnitude (mm)",
            kind="violin",
            inner="stick",
        )
        sf2.ax.set_yticklabels(sf2.ax.get_yticklabels(), rotation=0)
        sf2.fig.set_size_inches(6, 2)
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + "_2.png")
        if self.inputs.verbose:
            print(f"\t\t\t - Save report image 2 as {image_filename}...")
        sf2.savefig(image_filename, dpi=150)
        plt.close(sf2.fig)

        # Show the percentage of slice with NaNs for centroids.
        # It can occur when the brain mask does not cover the slice
        sf3 = sns.catplot(
            data=df, y="Scan", x="Proportion of NaNs (%)", kind="bar"
        )
        sf3.ax.set_yticklabels(sf3.ax.get_yticklabels(), rotation=0)
        sf3.fig.set_size_inches(6, 2)
        # Save the temporary report image
        image_filename = os.path.abspath(image_basename + "_3.png")
        if self.inputs.verbose:
            print(f"\t\t\t - Save report image 3 as {image_filename}...")
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
        axs.imshow(read_image(image_basename + "_0.png"))
        axs.set_axis_off()

        axs = subfigs.flat[1].subplots(3, 1)
        for i, ax in enumerate(axs):
            ax.imshow(read_image(image_basename + f"_{i+1}.png"))
            ax.set_axis_off()

        # Save the final report image
        image_filename = self._gen_filename("report_image")
        if self.inputs.verbose:
            print(f"\t\t\t - Save final report image as {image_filename}...")
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
            (
                score[f],
                prop_of_nans[f],
                centroid_coordx[f],
                centroid_coordy[f],
            ) = self._compute_motion_index(f)
            motion_ind.append(score[f])

        self._create_report_image(
            score, prop_of_nans, centroid_coordx, centroid_coordy
        )

        vp_defined = -1 not in [f.find("vp") for f in self.inputs.input_masks]
        if vp_defined:
            orientations_ = []
            for f in self.inputs.input_masks:
                orientations_.append((f.split("_vp-")[1]).split("_")[0])
            _, images_ordered, orientations_ordered = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(motion_ind, self.inputs.input_masks, orientations_)
                    )
                )
            )
        else:
            _, images_ordered = (
                list(t)
                for t in zip(*sorted(zip(motion_ind, self.inputs.input_masks)))
            )

        run_order = [
            int(f.split("run-")[1].split("_")[0]) for f in images_ordered
        ]

        if vp_defined:
            first_ax = orientations_ordered.index("ax")
            first_sag = orientations_ordered.index("sag")
            first_cor = orientations_ordered.index("cor")
            firsts = [first_ax, first_cor, first_sag]

            run_tmp = run_order
            run_order = []
            ind_ = firsts.index(min(firsts))
            run_order.append(
                int(
                    images_ordered[firsts[ind_]].split("run-")[1].split("_")[0]
                )
            )

            firsts.pop(ind_)
            ind_ = firsts.index(min(firsts))
            run_order.append(
                int(
                    images_ordered[firsts[ind_]].split("run-")[1].split("_")[0]
                )
            )

            firsts.pop(ind_)
            ind_ = firsts.index(min(firsts))
            run_order.append(
                int(
                    images_ordered[firsts[ind_]].split("run-")[1].split("_")[0]
                )
            )

            others = [e for e in run_tmp if e not in run_order]
            run_order += others

        return run_order


####################
# Brain Extraction
####################


class BrainExtractionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the BrainExtraction interface."""

    in_file = File(desc="Input image", mandatory=True)
    in_ckpt_loc = File(
        desc="Network_checkpoint for localization", mandatory=True
    )
    threshold_loc = traits.Float(
        0.49, desc="Threshold determining cutoff probability (0.49 by default)"
    )
    in_ckpt_seg = File(
        desc="Network_checkpoint for segmentation", mandatory=True
    )
    threshold_seg = traits.Float(
        0.5, desc="Threshold for cutoff probability (0.5 by default)"
    )
    out_postfix = traits.Str(
        "_brainMask",
        desc="Suffix of the automatically generated mask",
        usedefault=True,
    )


class BrainExtractionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the BrainExtraction interface."""

    out_file = File(desc="Output brain mask image")


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
        if name == "out_file":
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        self._extractBrain(
            self.inputs.in_file,
            self.inputs.in_ckpt_loc,
            self.inputs.threshold_loc,
            self.inputs.in_ckpt_seg,
            self.inputs.threshold_seg,
        )

        return runtime

    def _extractBrain(
        self, dataPath, modelCkptLoc, thresholdLoc, modelCkptSeg, thresholdSeg
    ):  # , bidsDir, out_postfix):
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
            from tflearn.layers.conv import (
                conv_2d,
                max_pool_2d,
                upsample_2d,
            )  # noqa: E402
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

        img_nib = nib.load(os.path.join(dataPath))
        image_data = img_nib.get_data()
        max_val = np.max(image_data)
        images = np.zeros((image_data.shape[2], width, height, n_channels))
        pred3dFinal = np.zeros(
            (
                image_data.shape[2],
                image_data.shape[0],
                image_data.shape[1],
                n_channels,
            )
        )

        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(
                image_data[:, :, ii],
                dsize=(width, height),
                fx=width,
                fy=height,
            )
            if normalize:
                if normalize == "local_max":
                    images[slice_counter, :, :, 0] = img_patch / np.max(
                        img_patch
                    )
                elif normalize == "global_max":
                    images[slice_counter, :, :, 0] = img_patch / max_val
                elif normalize == "mean_std":
                    images[slice_counter, :, :, 0] = (
                        img_patch - np.mean(img_patch)
                    ) / np.std(img_patch)
                else:
                    raise ValueError("Please select a valid normalization")
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

        g = tf.Graph()
        with g.as_default():

            with tf.name_scope("inputs"):
                x = tf.placeholder(
                    tf.float32, [None, width, height, n_channels], name="image"
                )

            conv1 = conv_2d(
                x, 32, 3, activation="relu", padding="same", regularizer="L2"
            )
            conv1 = conv_2d(
                conv1,
                32,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool1 = max_pool_2d(conv1, 2)

            conv2 = conv_2d(
                pool1,
                64,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv2 = conv_2d(
                conv2,
                64,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool2 = max_pool_2d(conv2, 2)

            conv3 = conv_2d(
                pool2,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv3 = conv_2d(
                conv3,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool3 = max_pool_2d(conv3, 2)

            conv4 = conv_2d(
                pool3,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv4 = conv_2d(
                conv4,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool4 = max_pool_2d(conv4, 2)

            conv5 = conv_2d(
                pool4,
                512,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv5 = conv_2d(
                conv5,
                512,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up6 = upsample_2d(conv5, 2)
            up6 = tflearn.layers.merge_ops.merge(
                [up6, conv4], "concat", axis=3
            )
            conv6 = conv_2d(
                up6,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv6 = conv_2d(
                conv6,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up7 = upsample_2d(conv6, 2)
            up7 = tflearn.layers.merge_ops.merge(
                [up7, conv3], "concat", axis=3
            )
            conv7 = conv_2d(
                up7,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv7 = conv_2d(
                conv7,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up8 = upsample_2d(conv7, 2)
            up8 = tflearn.layers.merge_ops.merge(
                [up8, conv2], "concat", axis=3
            )
            conv8 = conv_2d(
                up8, 64, 3, activation="relu", padding="same", regularizer="L2"
            )
            conv8 = conv_2d(
                conv8,
                64,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up9 = upsample_2d(conv8, 2)
            up9 = tflearn.layers.merge_ops.merge(
                [up9, conv1], "concat", axis=3
            )
            conv9 = conv_2d(
                up9, 32, 3, activation="relu", padding="same", regularizer="L2"
            )
            conv9 = conv_2d(
                conv9,
                32,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            pred = conv_2d(conv9, 2, 1, activation="linear", padding="valid")

        # Thresholding parameter to binarize predictions
        percentileLoc = thresholdLoc * 100

        pred3d = []
        with tf.Session(graph=g) as sess_test_loc:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test_loc, modelCkptLoc)

            for idx in range(images.shape[0]):

                im = np.reshape(
                    images[idx, :, :, :], [1, width, height, n_channels]
                )

                feed_dict = {x: im}
                pred_ = sess_test_loc.run(pred, feed_dict=feed_dict)

                theta = np.percentile(pred_, percentileLoc)
                pred_bin = np.where(pred_ > theta, 1, 0)
                pred3d.append(pred_bin[0, :, :, 0].astype("float64"))

            pred3d = np.asarray(pred3d)
            heights = []
            widths = []
            coms_x = []
            coms_y = []

            # Apply PPP
            ppp = True
            if ppp:
                pred3d = self._post_processing(pred3d)

            pred3d = [
                cv2.resize(
                    elem,
                    dsize=(image_data.shape[1], image_data.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                for elem in pred3d
            ]
            pred3d = np.asarray(pred3d)
            for i in range(np.asarray(pred3d).shape[0]):
                if np.sum(pred3d[i, :, :]) != 0:
                    pred3d[i, :, :] = self._extractLargestCC(
                        pred3d[i, :, :].astype("uint8")
                    )
                    contours, _ = cv2.findContours(
                        pred3d[i, :, :].astype("uint8"),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    area = cv2.minAreaRect(np.squeeze(contours))
                    heights.append(area[1][0])
                    widths.append(area[1][1])
                    bbox = cv2.boxPoints(area).astype("int")
                    coms_x.append(
                        int((np.max(bbox[:, 1]) + np.min(bbox[:, 1])) / 2)
                    )
                    coms_y.append(
                        int((np.max(bbox[:, 0]) + np.min(bbox[:, 0])) / 2)
                    )
            # Saving localization points
            med_x = int(np.median(coms_x))
            med_y = int(np.median(coms_y))
            half_max_x = int(np.max(heights) / 2)
            half_max_y = int(np.max(widths) / 2)
            x_beg = med_x - half_max_x - border_x
            x_end = med_x + half_max_x + border_x
            y_beg = med_y - half_max_y - border_y
            y_end = med_y + half_max_y + border_y

        # Step 2: Brain segmentation
        width = 96
        height = 96

        images = np.zeros((image_data.shape[2], width, height, n_channels))

        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(
                image_data[x_beg:x_end, y_beg:y_end, ii], dsize=(width, height)
            )

            if normalize:
                if normalize == "local_max":
                    images[slice_counter, :, :, 0] = img_patch / np.max(
                        img_patch
                    )
                elif normalize == "mean_std":
                    images[slice_counter, :, :, 0] = (
                        img_patch - np.mean(img_patch)
                    ) / np.std(img_patch)
                else:
                    raise ValueError("Please select a valid normalization")
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

        g = tf.Graph()
        with g.as_default():

            with tf.name_scope("inputs"):
                x = tf.placeholder(
                    tf.float32, [None, width, height, n_channels]
                )

            conv1 = conv_2d(
                x, 32, 3, activation="relu", padding="same", regularizer="L2"
            )
            conv1 = conv_2d(
                conv1,
                32,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool1 = max_pool_2d(conv1, 2)

            conv2 = conv_2d(
                pool1,
                64,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv2 = conv_2d(
                conv2,
                64,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool2 = max_pool_2d(conv2, 2)

            conv3 = conv_2d(
                pool2,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv3 = conv_2d(
                conv3,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool3 = max_pool_2d(conv3, 2)

            conv4 = conv_2d(
                pool3,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv4 = conv_2d(
                conv4,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            pool4 = max_pool_2d(conv4, 2)

            conv5 = conv_2d(
                pool4,
                512,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv5 = conv_2d(
                conv5,
                512,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up6 = upsample_2d(conv5, 2)
            up6 = tflearn.layers.merge_ops.merge(
                [up6, conv4], "concat", axis=3
            )
            conv6 = conv_2d(
                up6,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv6 = conv_2d(
                conv6,
                256,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up7 = upsample_2d(conv6, 2)
            up7 = tflearn.layers.merge_ops.merge(
                [up7, conv3], "concat", axis=3
            )
            conv7 = conv_2d(
                up7,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )
            conv7 = conv_2d(
                conv7,
                128,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up8 = upsample_2d(conv7, 2)
            up8 = tflearn.layers.merge_ops.merge(
                [up8, conv2], "concat", axis=3
            )
            conv8 = conv_2d(
                up8, 64, 3, activation="relu", padding="same", regularizer="L2"
            )
            conv8 = conv_2d(
                conv8,
                64,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            up9 = upsample_2d(conv8, 2)
            up9 = tflearn.layers.merge_ops.merge(
                [up9, conv1], "concat", axis=3
            )
            conv9 = conv_2d(
                up9, 32, 3, activation="relu", padding="same", regularizer="L2"
            )
            conv9 = conv_2d(
                conv9,
                32,
                3,
                activation="relu",
                padding="same",
                regularizer="L2",
            )

            pred = conv_2d(conv9, 2, 1, activation="linear", padding="valid")

        with tf.Session(graph=g) as sess_test_seg:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test_seg, modelCkptSeg)

            for idx in range(images.shape[0]):
                im = np.reshape(
                    images[idx, :, :], [1, width, height, n_channels]
                )
                feed_dict = {x: im}
                pred_ = sess_test_seg.run(pred, feed_dict=feed_dict)
                percentileSeg = thresholdSeg * 100
                theta = np.percentile(pred_, percentileSeg)
                pred_bin = np.where(pred_ > theta, 1, 0)
                # Map predictions to original indices and size
                pred_bin = cv2.resize(
                    pred_bin[0, :, :, 0],
                    dsize=(y_end - y_beg, x_end - x_beg),
                    interpolation=cv2.INTER_NEAREST,
                )
                pred3dFinal[
                    idx, x_beg:x_end, y_beg:y_end, 0
                ] = pred_bin.astype("float64")

            pppp = True
            if pppp:
                pred3dFinal = self._post_processing(np.asarray(pred3dFinal))
            pred3d = [
                cv2.resize(
                    elem,
                    dsize=(image_data.shape[1], image_data.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                for elem in pred3dFinal
            ]
            pred3d = np.asarray(pred3d)
            upsampled = np.swapaxes(
                np.swapaxes(pred3d, 1, 2), 0, 2
            )  # if Orient module applied, no need for this line(?)
            up_mask = nib.Nifti1Image(upsampled, img_nib.affine)

            # Save output mask
            save_file = self._gen_filename("out_file")
            nib.save(up_mask, save_file)

    def _extractLargestCC(self, image):
        """Function returning largest connected component of an object."""

        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
            image, connectivity=4
        )
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
        return largest_cc.astype("uint8")

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
                        crt_stack_cc[
                            wherr[0][ii], wherr[1][ii], wherr[2][ii]
                        ] = 0

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
                local_minima = argrelextrema(np.asarray(distrib_cc), np.less)[
                    0
                ]
                local_maxima = argrelextrema(
                    np.asarray(distrib_cc), np.greater
                )[0]

                for iMin, _ in enumerate(local_minima):
                    for iMax in range(len(local_maxima) - 1):
                        # find between which maxima is the minima localized
                        if (
                            local_maxima[iMax]
                            < local_minima[iMin]
                            < local_maxima[iMax + 1]
                        ):
                            # check if diff max-min is large enough to be considered
                            if (
                                distrib_cc[local_maxima[iMax]]
                                - distrib_cc[local_minima[iMin]]
                                > 50
                            ) and (
                                distrib_cc[local_maxima[iMax + 1]]
                                - distrib_cc[local_minima[iMin]]
                                > 50
                            ):
                                sub_stack = crt_stack_closed_minima[
                                    local_maxima[iMax]
                                    - 1 : local_maxima[iMax + 1]
                                    + 1,
                                    :,
                                    :,
                                ]
                                sub_stack = binary_closing(sub_stack)
                                crt_stack_closed_minima[
                                    local_maxima[iMax]
                                    - 1 : local_maxima[iMax + 1]
                                    + 1,
                                    :,
                                    :,
                                ] = sub_stack
                crt_stack_pp = crt_stack_closed_minima.copy()

                distrib_closed = []
                for iSlc in range(crt_stack_closed_minima.shape[0]):
                    distrib_closed.append(
                        np.sum(crt_stack_closed_minima[iSlc])
                    )

            if post_proc_opening_maxima:
                crt_stack_opened_maxima = crt_stack_pp.copy()

                local = True
                if local:
                    local_maxima_n = argrelextrema(
                        np.asarray(distrib_closed), np.greater
                    )[
                        0
                    ]  # default is mode='clip'. Doesn't consider extremity as being an extrema

                    for iMax, _ in enumerate(local_maxima_n):
                        # Check if this local maxima is a "peak"
                        if (
                            distrib[local_maxima_n[iMax]]
                            - distrib[local_maxima_n[iMax] - 1]
                            > 50
                        ) and (
                            distrib[local_maxima_n[iMax]]
                            - distrib[local_maxima_n[iMax] + 1]
                            > 50
                        ):

                            if verbose:
                                print(
                                    "Ceci est un pic de au moins 50.",
                                    distrib[local_maxima_n[iMax]],
                                    "en",
                                    local_maxima_n[iMax],
                                )
                                print(
                                    "                                bordé de",
                                    distrib[local_maxima_n[iMax] - 1],
                                    "en",
                                    local_maxima_n[iMax] - 1,
                                )
                                print(
                                    "                                et",
                                    distrib[local_maxima_n[iMax] + 1],
                                    "en",
                                    local_maxima_n[iMax] + 1,
                                )
                                print("")

                            sub_stack = crt_stack_opened_maxima[
                                local_maxima_n[iMax]
                                - 1 : local_maxima_n[iMax]
                                + 2,
                                :,
                                :,
                            ]
                            sub_stack = binary_opening(sub_stack)
                            crt_stack_opened_maxima[
                                local_maxima_n[iMax]
                                - 1 : local_maxima_n[iMax]
                                + 2,
                                :,
                                :,
                            ] = sub_stack
                else:
                    crt_stack_opened_maxima = binary_opening(
                        crt_stack_opened_maxima
                    )
                crt_stack_pp = crt_stack_opened_maxima.copy()

                distrib_opened = []
                for iSlc in range(crt_stack_pp.shape[0]):
                    distrib_opened.append(np.sum(crt_stack_pp[iSlc]))

            if post_proc_extremity:
                crt_stack_extremity = crt_stack_pp.copy()

                # check si y a un maxima sur une extremite
                maxima_extrema = argrelextrema(
                    np.asarray(distrib_closed), np.greater, mode="wrap"
                )[0]

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
        outputs["out_file"] = self._gen_filename("out_file")
        return outputs


class ReduceFieldOfViewInputSpec(BaseInterfaceInputSpec):
    """Class."""

    input_image = File(mandatory=True, desc="Input image filename")
    input_mask = File(mandatory=True, desc="Input mask filename")
    input_label = File(mandatory=False, desc="Input label filename")


class ReduceFieldOfViewOutputSpec(TraitedSpec):
    """Class"""

    output_image = File(desc="Cropped image")
    output_mask = File(desc="Cropped mask")
    output_label = File(desc="Cropped labels")


class ReduceFieldOfView(BaseInterface):
    """Runs the"""

    input_spec = ReduceFieldOfViewInputSpec
    output_spec = ReduceFieldOfViewOutputSpec

    def _gen_filename(self, name):
        if name == "output_image":
            return os.path.abspath(os.path.basename(self.inputs.input_image))
        elif name == "output_mask":
            return os.path.abspath(os.path.basename(self.inputs.input_mask))
        elif name == "output_label":
            return os.path.abspath(os.path.basename(self.inputs.input_label))
        return None

    def _crop_image_and_mask(
        self, in_image, in_mask, in_label, paddings_mm=[15, 15, 15]
    ):
        import SimpleITK as sitk

        reader = sitk.ImageFileReader()

        reader.SetFileName(in_mask)
        mask = reader.Execute()
        mask_np = sitk.GetArrayFromImage(mask)

        reader.SetFileName(in_image)
        image = reader.Execute()
        image_np = sitk.GetArrayFromImage(image)

        im_shape = list(image_np.shape)

        # Compute ROI bounding box
        minimums = [0, 0, 0]
        maximums = list(mask_np.shape)
        ri = skimage.measure.regionprops(
            (mask_np > 0).astype(np.uint8), image_np
        )
        (
            minimums[0],
            minimums[1],
            minimums[2],
            maximums[0],
            maximums[1],
            maximums[2],
        ) = ri[0].bbox

        # Convert padding from mm to voxels
        paddings = [0, 0, 0]
        resolutions = list(image.GetSpacing())
        resolutions.reverse()
        for i in range(3):
            paddings[i] = int(np.round(paddings_mm[i] / resolutions[i]))

        # Update ROI bounding box with padding
        for i in range(3):
            minimums[i] = int(max(0, minimums[i] - paddings[i]))
            maximums[i] = int(min(im_shape[i], maximums[i] + paddings[i]))

        # Crop ROI FOV
        image_np = image_np[
            minimums[0] : maximums[0],
            minimums[1] : maximums[1],
            minimums[2] : maximums[2],
        ]
        mask_np = mask_np[
            minimums[0] : maximums[0],
            minimums[1] : maximums[1],
            minimums[2] : maximums[2],
        ]

        minimums_copy = minimums.copy()
        minimums_copy.reverse()

        new_origin = list(
            image.TransformContinuousIndexToPhysicalPoint(minimums_copy)
        )

        new_direction = list(image.GetDirection())
        new_spacing = list(image.GetSpacing())

        image_cropped = sitk.GetImageFromArray(image_np)
        image_cropped.SetOrigin(new_origin)
        image_cropped.SetDirection(new_direction)
        image_cropped.SetSpacing(new_spacing)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(self._gen_filename("output_image"))
        writer.Execute(image_cropped)

        mask_cropped = sitk.GetImageFromArray(mask_np)
        mask_cropped.SetOrigin(new_origin)
        mask_cropped.SetDirection(new_direction)
        mask_cropped.SetSpacing(new_spacing)

        writer = sitk.ImageFileWriter()
        writer.SetFileName(self._gen_filename("output_mask"))
        writer.Execute(mask_cropped)

        if in_label:
            reader.SetFileName(in_label)
            label = reader.Execute()
            label_np = sitk.GetArrayFromImage(label)

            label_np = label_np[
                minimums[0] : maximums[0],
                minimums[1] : maximums[1],
                minimums[2] : maximums[2],
            ]

            label_cropped = sitk.GetImageFromArray(label_np)
            label_cropped.SetOrigin(new_origin)
            label_cropped.SetDirection(new_direction)
            label_cropped.SetSpacing(new_spacing)

            writer = sitk.ImageFileWriter()
            writer.SetFileName(self._gen_filename("output_label"))
            writer.Execute(label_cropped)

    def _run_interface(self, runtime):

        self._crop_image_and_mask(
            self.inputs.input_image,
            self.inputs.input_mask,
            self.inputs.input_label,
        )

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_image"] = self._gen_filename("output_image")
        outputs["output_mask"] = self._gen_filename("output_mask")
        if self.inputs.input_label:
            outputs["output_label"] = self._gen_filename("output_label")
        return outputs


class SplitLabelMapsInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the SplitLabelMaps interface."""

    in_labelmap = File(desc="Input label map", mandatory=True)
    all_labels = traits.List([], mandatory=False)


class SplitLabelMapsOutputSpec(TraitedSpec):
    """Class used to represent outputs of the SplitLabelMaps interface."""

    out_labelmaps = OutputMultiPath(File(), desc="Output masks")
    out_labels = traits.List(desc="List of labels ids that were extracted")


class SplitLabelMaps(BaseInterface):
    """Split a multi-label labelmap
    into one label map per label.
    """

    input_spec = SplitLabelMapsInputSpec
    output_spec = SplitLabelMapsOutputSpec

    _labels = None

    def _gen_filename(self, name, i):
        if name == "out_label":
            _, name, ext = split_filename(self.inputs.in_labelmap)
            if "labels" in name:
                output = name.replace("labels", "label-" + str(i)) + ext
            return os.path.abspath(output)
        return None

    def _extractlabelimage(self, in_labelmap):
        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(in_labelmap)
        labels = reader.Execute()

        binarizer = sitk.BinaryThresholdImageFilter()

        if not len(self.inputs.all_labels):
            self._labels = list(
                np.unique(sitk.GetArrayFromImage(labels)).astype(int)
            )
        else:
            self._labels = self.inputs.all_labels

        for label_id in self._labels:
            binarizer.SetLowerThreshold(int(label_id))
            binarizer.SetUpperThreshold(int(label_id))

            label = binarizer.Execute(labels)

            writer.SetFileName(self._gen_filename("out_label", label_id))
            writer.Execute(label)

    def _run_interface(self, runtime):

        self._extractlabelimage(self.inputs.in_labelmap)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_labelmaps"] = [
            self._gen_filename("out_label", i) for i in self._labels
        ]
        outputs["out_labels"] = self._labels
        return outputs


class ListsMergerInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the PathListsMerger interface."""

    inputs = traits.List()


class ListsMergerOutputSpec(TraitedSpec):
    """Class used to represent outputs of the PathListsMerger interface."""

    outputs = traits.List()


class ListsMerger(BaseInterface):
    """Interface to merge list of paths or list of list of path"""

    input_spec = ListsMergerInputSpec
    output_spec = ListsMergerOutputSpec

    m_list_of_files = None

    def _gen_filename(self, name):
        if name == "outputs":
            return self.m_list_of_files
        return None

    def _run_interface(self, runtime):
        self.m_list_of_files = []
        for list_of_one_stack in self.inputs.inputs:
            if isinstance(list_of_one_stack, list) or isinstance(
                list_of_one_stack, InputMultiPath
            ):
                for file in list_of_one_stack:
                    self.m_list_of_files.append(file)
            else:
                self.m_list_of_files.append(list_of_one_stack)

        self.m_list_of_files = list(set(self.m_list_of_files))

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["outputs"] = self._gen_filename("outputs")
        return outputs


class ResampleImageInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the ResampleImage interface."""

    input_image = File(mandatory=True, desc="Input image to resample")
    input_reference = File(
        mandatory=True, desc="Input image with reference resolution"
    )
    verbose = traits.Bool(desc="Enable verbosity")


class ResampleImageOutputSpec(TraitedSpec):
    """Class used to represent outputs of the ResampleImage interface."""

    output_image = File(desc="Masked image")


class ResampleImage(BaseInterface):
    """Retrieve atlas of the same age and
    resample it to subject's in-plane resolution
    """

    input_spec = ResampleImageInputSpec
    output_spec = ResampleImageOutputSpec

    def _gen_filename(self, name):
        if name == "output_image":
            return os.path.abspath(os.path.basename(self.inputs.input_image))
        return None

    def _run_interface(self, runtime):

        target_resolution = self._get_target_resolution(
            reference_image=self.inputs.input_reference
        )
        self._resample_image(
            p_image_path=self.inputs.input_image,
            p_resolution=target_resolution,
        )

        return runtime

    def _get_target_resolution(self, reference_image):
        reader = sitk.ImageFileReader()
        reader.SetFileName(reference_image)
        sub_image = reader.Execute()

        spacings = list(sub_image.GetSpacing())
        spacings.sort()
        if self.inputs.verbose:
            print("Target isotropic spacing:", spacings[0])
        return spacings[0]

    def _resample_image(self, p_image_path, p_resolution):

        ants_path = "/opt/conda/bin"

        image_resampled_path = self._gen_filename("output_image")

        cmd = (
            f"ResampleImageBySpacing 3 {p_image_path} "
            f"{image_resampled_path} {str(p_resolution)} "
            f"{str(p_resolution)} {str(p_resolution)}"
        )
        if self.inputs.verbose:
            print("\n\n" + cmd + "\n\n")
        run(cmd, env={"PATH": ants_path})

        cmd = (
            f"antsApplyTransforms -d 3 -i {p_image_path} "
            f"-r {image_resampled_path} -o {image_resampled_path} "
            f"-t [identity]"
        )
        run(cmd, env={"PATH": ants_path})
        if self.inputs.verbose:
            print("Reference STA was resampled.")

        return image_resampled_path

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_image"] = self._gen_filename("output_image")
        return outputs


class ComputeAlignmentToReferenceInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the
    ComputeAlignmentToReference interface."""

    input_image = File(mandatory=True, desc="Input image to realign")
    input_template = File(mandatory=True, desc="Input reference image")


class ComputeAlignmentToReferenceOutputSpec(TraitedSpec):
    """Class used to represent outputs of the
    ComputeAlignmentToReference interface."""

    output_transform = File(
        mandatory=True, desc="Output 3D rigid tranformation file"
    )


class ComputeAlignmentToReference(BaseInterface):
    """Reorient image along reference, based on principal brain axis.

    This module relies on the implementation [1]_ from EbnerWang2020 [2]_.

    References
    ------------
    .. [1] `(link to github) <https://github.com/gift-surg/NiftyMIC>`_
    .. [2] Ebner et al. (2020). An automated framework for localization,
    segmentation and super-resolution reconstruction of fetal brain MRI.
     NeuroImage, 206, 116324. `(link to paper)
     <https://www.sciencedirect.com/science/article/pii/S1053811919309152>`_

    Examples
    --------
    >>>

    """

    input_spec = ComputeAlignmentToReferenceInputSpec
    output_spec = ComputeAlignmentToReferenceOutputSpec

    m_best_transform = None

    def _gen_filename(self, name, i_o=-1):
        _, basename, _ = split_filename(self.inputs.input_image)
        if name == "output_transform":
            output = basename + "_rigid" + ".tfm"
            return os.path.abspath(output)
        elif name == "output_image":
            output = basename + "_reoriented" + "_" + str(i_o) + ".nii.gz"
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        self.m_best_transform = self._reorient_image()

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_transform"] = self._gen_filename("output_transform")
        return outputs

    def _compute_pca(self, mask):
        def get_largest_connected_region_mask(mask_nda):
            """This function is from:
            https://github.com/gift-surg/NiftyMIC/blob/e62c5389dfa2bb367fb217b7060472978d3e7654/niftymic/utilities/template_stack_estimator.py#L123
            """
            # get label for each connected component
            labels_nda = skimage.measure.label(mask_nda)

            # only pick largest connected region
            if labels_nda.max() > 1:
                volumes = [
                    labels_nda[np.where(labels_nda == i)].sum()
                    for i in range(1, labels_nda.max() + 1)
                ]
                label_max = np.argmax(np.array(volumes)) + 1
                mask_nda = np.zeros_like(mask_nda)
                mask_nda[np.where(labels_nda == label_max)] = 1

            return mask_nda

        mask_nda = sitk.GetArrayFromImage(mask)
        mask_nda = mask_nda > 0

        # # We do a closing (dilation+erosion),
        # in case slices are discarded from stacks
        # closing = sitk.BinaryMorphologicalClosingImageFilter()
        # mask_nda = closing.Execute(mask_nda)

        # # get largest connected region (if more than one connected region)
        mask_nda = get_largest_connected_region_mask(mask_nda)

        # [z, y, x] x n_points to [x, y, z] x n_points
        points = np.array(np.where(mask_nda > 0))[::-1, :]
        n_points = len(points[0])
        for i in range(n_points):
            points[:, i] = mask.TransformIndexToPhysicalPoint(
                [int(j) for j in points[:, i]]
            )

        pca_fixed = pca.PrincipalComponentAnalysis(points.transpose())

        pca_fixed.run()
        return pca_fixed

    def _reorient_image(self):
        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(self.inputs.input_image)
        sub = reader.Execute()

        reader.SetFileName(self.inputs.input_template)
        template = reader.Execute()

        # - PBA computation
        pca_fixed = self._compute_pca(template)
        pca_moving = self._compute_pca(sub)

        # perform PCAs for fixed and moving images
        eigvec_moving = pca_moving.get_eigvec()
        mean_moving = pca_moving.get_mean()

        eigvec_fixed = pca_fixed.get_eigvec()
        mean_fixed = pca_fixed.get_mean()

        # test different initializations based on eigenvector orientations
        orientations = [
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
        ]

        transformations = []
        similarities_abs = []

        for i_o, orientation in enumerate(orientations):
            eigvec_moving_o = np.array(eigvec_moving)
            eigvec_moving_o[:, 0] *= orientation[0]
            eigvec_moving_o[:, 1] *= orientation[1]

            # get right-handed coordinate system
            cross = np.cross(eigvec_moving_o[:, 0], eigvec_moving_o[:, 1])
            eigvec_moving_o[:, 2] = cross

            # transformation to align fixed with moving eigenbasis
            R = eigvec_moving_o.dot(eigvec_fixed.transpose())
            t = mean_moving - R.dot(mean_fixed)

            # build rigid transformation as sitk object
            rigid_transform_sitk = sitk.VersorRigid3DTransform()
            rigid_transform_sitk.SetMatrix(R.flatten())
            rigid_transform_sitk.SetTranslation(t)
            transformations.append(rigid_transform_sitk)

            warped_moving_sitk_sta = sitk.Resample(
                sub,
                template,
                rigid_transform_sitk,
                sitk.sitkLinear,  # Reference
            )

            im_tfm = self._gen_filename("output_image", i_o)
            writer.SetFileName(im_tfm)
            writer.Execute(warped_moving_sitk_sta)

            similarity = Similarity()
            similarity.inputs.volume1 = self.inputs.input_template
            similarity.inputs.volume2 = im_tfm
            similarity.inputs.metric = "mi"

            mi = similarity.run()
            similarities_abs.append(abs(float(mi.outputs.similarity[0])))

        i_best_transform = similarities_abs.index(max(similarities_abs))
        best_transform = transformations[i_best_transform]

        sitk.WriteTransform(
            best_transform.GetInverse(), self._gen_filename("output_transform")
        )

        return i_best_transform


class ApplyAlignmentTransformInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the
    ApplyAlignmentTransform interface."""

    input_image = File(mandatory=True, desc="Input image to realign")
    input_template = File(mandatory=True, desc="Input reference image")

    input_mask = File(mandatory=False, desc="Input mask to realign")

    input_transform = File(
        mandatory=True, desc="Input alignment transform to apply"
    )


class ApplyAlignmentTransformOutputSpec(TraitedSpec):
    """Class used to represent outputs of the
    ApplyAlignmentTransform interface."""

    output_image = File(mandatory=True, desc="Output reoriented image")
    output_mask = File(mandatory=False, desc="Output reoriented mask")


class ApplyAlignmentTransform(BaseInterface):
    """Apply a rigid 3D transform.

    Examples
    --------
    >>>

    """

    input_spec = ApplyAlignmentTransformInputSpec
    output_spec = ApplyAlignmentTransformOutputSpec

    def _gen_filename(self, name):
        if name == "output_image":
            output = os.path.basename(self.inputs.input_image)
            return os.path.abspath(output)
        elif name == "output_mask":
            output = os.path.basename(self.inputs.input_mask)
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):
        self._reorient_image()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_image"] = self._gen_filename("output_image")
        outputs["output_mask"] = self._gen_filename("output_mask")
        return outputs

    def _reorient_image(self):
        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(self.inputs.input_image)
        sub = reader.Execute()

        if self.inputs.input_mask:
            reader.SetFileName(self.inputs.input_mask)
            mask = reader.Execute()

        reader.SetFileName(self.inputs.input_template)
        template = reader.Execute()

        # build rigid transformation as sitk VersorRigid3DTransform object
        transform = sitk.ReadTransform(self.inputs.input_transform)
        transform_params = transform.GetParameters()

        rigid_transform_sitk = sitk.VersorRigid3DTransform()
        rigid_transform_sitk.SetParameters(transform_params)
        rigid_transform_sitk = rigid_transform_sitk.GetInverse()

        warped_moving_sitk_sta = sitk.Resample(
            sub, template, rigid_transform_sitk, sitk.sitkLinear  # Reference
        )

        im_tfm = self._gen_filename("output_image")
        writer.SetFileName(im_tfm)
        writer.Execute(warped_moving_sitk_sta)

        if self.inputs.input_mask:
            warped_moving_sitk_mask = sitk.Resample(
                mask,
                template,  # Reference
                rigid_transform_sitk,
                sitk.sitkNearestNeighbor,
            )

            mask_tfm = self._gen_filename("output_mask")
            writer.SetFileName(mask_tfm)
            writer.Execute(warped_moving_sitk_mask)

        return
