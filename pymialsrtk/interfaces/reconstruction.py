# Copyright © 2016-2019 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK preprocessing functions
"""

import os

from glob import glob

from traits.api import *

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

from pymialsrtk.interfaces.utils import run


########################
# Image Reconstruction
########################

class MialsrtkImageReconstructionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkImageReconstruction interface.

    Attributes
    ----------
    bids_dir <string>
        BIDS root directory (required)

    input_images <list<string>>
        Input image filenames (required)

    input_masks <list<string>>
        Mask of the input images (required)

    in_roi <enum>
        Define region of interest (required):
            - 'box': Use intersections for roi calculation
            - 'mask': Use masks for roi calculation
            - 'all': Use the whole image FOV

    input_rad_dilatation <float>
        Radius dilatation used in prior step to construct output filename. (default is 1.0)

    sub_ses <string>
        Subject and session BIDS identifier to construct output filename.

    out_sdi_postfix <string>
        suffix added to construct output scattered data interpolation filename (default is '_SDI')

    out_transf_postfix <string>
        suffix added to construct output transformation filenames (default is '_transform')

    stacks_order <list<int>>
        order of images index. To ensure images are processed with their correct corresponding mask.

    See Also
    ----------
    pymialsrtk.interfaces.preprocess.MialsrtkImageReconstruction
    """
    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    in_roi = traits.Enum('mask', "all", "box", "mask", mandatory=True, usedefault=True)
    input_masks = InputMultiPath(File(desc='Input masks'))
    input_images = InputMultiPath(File(desc='Input images'))
    input_rad_dilatation = traits.Float(1.0, usedefault=True)
    sub_ses = traits.Str("x", usedefault=True)
    out_sdi_prefix = traits.Str("SDI_", usedefault=True)
    out_transf_postfix = traits.Str("_transform", usedefault=True)
    stacks_order = traits.List(mandatory=True)

class MialsrtkImageReconstructionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkImageReconstruction interface.

    Attributes
    -----------
    output_sdi <string>
        Output scattered data interpolation image file

    output_transforms <string>
        Output transformation files

    See also
    --------------
    pymialsrtk.interfaces.preprocess.MialsrtkImageReconstruction
    """
    output_sdi = File(desc='Output reconstructed image')
    output_transforms = OutputMultiPath(File(desc='Output transformation files'))


class MialsrtkImageReconstruction(BaseInterface):
    """Creates a high resolution image from a set of low resolution images [1]_.

    References
    ------------
    .. [1] Tourbier et al.; NeuroImage, 2015. `(link to paper) <https://doi.org/10.1016/j.neuroimage.2015.06.018>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.reconstruction import MialsrtkImageReconstruction
    >>> srtkImageReconstruction = MialsrtkTVSuperResolution()
    >>> srtkImageReconstruction.inputs.bids_dir = '/my_directory'
    >>> srtkImageReconstruction.input_images = ['image01.nii.gz', 'image02.nii.gz', 'image03.nii.gz', 'image04.nii.gz']
    >>> srtkImageReconstruction.input_masks = ['mask01.nii.gz', 'mask02.nii.gz', 'mask03.nii.gz', 'mask04.nii.gz']
    >>> srtkImageReconstruction.inputs.stacksOrder = [0,1,2,3]
    >>> srtkImageReconstruction.inputs.sub_ses = 'sub-01_ses-01'
    >>> srtkImageReconstruction.inputs.in_roi = 'mask'
    >>> srtkImageReconstruction.inputs.in_deltat = 0.01
    >>> srtkImageReconstruction.inputs.in_lambda = 0.75
    >>> srtkImageReconstruction.run()  # doctest: +SKIP
    """
    input_spec = MialsrtkImageReconstructionInputSpec
    output_spec = MialsrtkImageReconstructionOutputSpec

    def _run_interface(self, runtime):

        params = []
        params.append(''.join(["--", self.inputs.in_roi]))

        run_nb_images = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        run_nb_masks = []
        for in_mask in self.inputs.input_masks:
            cut_avt = in_mask.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_masks.append(int(cut_apr))

        for order in self.inputs.stacks_order:
            index_img = run_nb_images.index(order)

            _, name, ext = split_filename(os.path.abspath(self.inputs.input_images[index_img]))
            transf_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                       ''.join([name, self.inputs.out_transf_postfix, '_',
                                                str(len(self.inputs.stacks_order)), 'V', '.txt']))

            params.append("-i")
            params.append(self.inputs.input_images[index_img])

            if self.inputs.in_roi == "mask":
                index_mask = run_nb_masks.index(order)

                params.append("-m")
                params.append(self.inputs.input_masks[index_mask])

            params.append("-t")
            params.append(transf_file)

        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                ''.join(([self.inputs.out_sdi_prefix, self.inputs.sub_ses, '_',
                                          str(len(self.inputs.stacks_order)), 'V_rad',
                                          str(int(self.inputs.input_rad_dilatation)), ext])))

        params.append("-o")
        params.append(out_file)

        cmd = ["mialsrtkImageReconstruction"]
        cmd += params

        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            print("")
            print(cmd)
            print("")
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_transforms'] = glob(os.path.abspath("*.txt"))

        _, _, ext = split_filename(os.path.abspath(self.inputs.input_images[0]))

        outputs['output_sdi'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                             ''.join(([self.inputs.out_sdi_prefix, self.inputs.sub_ses, '_',
                                                       str(len(self.inputs.stacks_order)),'V_rad', str(int(self.inputs.input_rad_dilatation)), ext])))

        return outputs


#####################################
#  Total Variation Super Resolution
#####################################

class MialsrtkTVSuperResolutionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkTVSuperResolution interface.

    Attributes
    ----------
    bids_dir <string>
        BIDS root directory (required)

    input_images <list<string>>
        Input image filenames (required)

    input_masks <list<string>>
        Mask of the input images (required)

    input_transforms <list<string>>
        Input transformation filenames (required)

    input_sdi <string>
        Reconstructed image for initialization. Typically the output of MialsrtkImageReconstruction is used. (required)

    deblurring <bool>
        Flag to set deblurring PSF during SR (double the neighborhood) (default is 0).

    in_loop <int>
        Number of loops (SR/denoising) (required)

    in_deltat <float>
        Parameter deltat (required)

    in_lambda <float>
        Regularization factor (required)

    in_bregman_loop <int>
        Number of Bregman loops (default is 1)

    in_iter <int>
        Number of inner iterations (default is 50)

    in_step_scale <float>
        Parameter step scale (default is 10.0)

    in_gamma <float>
        Parameter gamma (default is 10.0)

    in_inner_thresh <float>
        Inner loop convergence threshold (default = 1e-5)

    in_outer_thresh <float>
        Outer loop convergence threshold (default = 1e-6)

    out_prefix <string>
        prefix added to construct output super-resolution filename (default is 'SRTV_')

    stacks_order <list<int>>
        order of images index. To ensure images are processed with their correct corresponding mask.

    input_rad_dilatation <float>
        Radius dilatation used in prior step to construct output filename. (default is 1.0)

    sub_ses <string>
        Subject and session BIDS identifier to construct output filename.

    See Also
    ----------
    pymialsrtk.interfaces.preprocess.MialsrtkTVSuperResolution
    """
    bids_dir = Directory(desc='BIDS root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(desc='files to be SR', mandatory=True))
    input_masks = InputMultiPath(File(desc='mask of files to be SR', mandatory=True))
    input_transforms = InputMultiPath(File(desc='', mandatory=True))
    input_sdi = File(File(desc='', mandatory=True))
    deblurring = traits.Bool(False, usedefault=True)

    in_loop = traits.Int(mandatory=True)
    in_deltat = traits.Float(mandatory=True)
    in_lambda = traits.Float(mandatory=True)

    in_bregman_loop = traits.Int(1, usedefault=True)
    in_iter = traits.Int(50, usedefault=True)
    in_step_scale = traits.Int(10, usedefault=True)
    in_gamma = traits.Int(10, usedefault=True)
    in_inner_thresh = traits.Float(0.00001, usedefault=True)
    in_outer_thresh = traits.Float(0.000001, usedefault=True)

    out_prefix = traits.Str("SRTV_", usedefault=True)
    stacks_order = traits.List(mandatory=False)

    input_rad_dilatation = traits.Float(1.0, usedefault=True)

    sub_ses = traits.Str("x", usedefault=True)


class MialsrtkTVSuperResolutionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkTVSuperResolution interface.

    Attributes
    -----------
    output_sr <string>
        Output super-resolution reconstruction file

    See also
    --------------
    pymialsrtk.interfaces.preprocess.MialsrtkTVSuperResolution
    """
    output_sr = File(desc='Super-resolution reconstruction output')


class MialsrtkTVSuperResolution(BaseInterface):
    """Apply super-resolution algorithm using one or multiple input images [1]_.

    References
    ------------
    .. [1] Tourbier et al.; NeuroImage, 2015. `(link to paper) <https://doi.org/10.1016/j.neuroimage.2015.06.018>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.reconstruction import MialsrtkTVSuperResolution
    >>> srtkTVSuperResolution = MialsrtkTVSuperResolution()
    >>> srtkTVSuperResolution.inputs.bids_dir = '/my_directory'
    >>> srtkTVSuperResolution.input_images = ['image01.nii.gz', 'image02.nii.gz', 'image03.nii.gz', 'image04.nii.gz']
    >>> srtkTVSuperResolution.input_masks = ['mask01.nii.gz', 'mask02.nii.gz', 'mask03.nii.gz', 'mask04.nii.gz']
    >>> srtkTVSuperResolution.input_transforms = ['transform01.txt', 'transform02.txt', 'transform03.txt', 'transform04.txt']
    >>> srtkTVSuperResolution.input_sdi = 'sdi.nii.gz'
    >>> srtkTVSuperResolution.inputs.stacksOrder = [0,1,2,3]
    >>> srtkTVSuperResolution.inputs.sub_ses = 'sub-01_ses-01'
    >>> srtkTVSuperResolution.inputs.in_loop = 10
    >>> srtkTVSuperResolution.inputs.in_deltat = 0.01
    >>> srtkTVSuperResolution.inputs.in_lambda = 0.75
    >>> srtkTVSuperResolution.run()  # doctest: +SKIP
    """

    input_spec = MialsrtkTVSuperResolutionInputSpec
    output_spec = MialsrtkTVSuperResolutionOutputSpec

    def _run_interface(self, runtime):

        cmd = ['mialsrtkTVSuperResolution']

        run_nb_images = []
        for in_file in self.inputs.input_images:
            cut_avt = in_file.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_images.append(int(cut_apr))

        run_nb_masks = []
        for in_mask in self.inputs.input_masks:
            cut_avt = in_mask.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_masks.append(int(cut_apr))

        run_nb_transforms = []
        for in_transform in self.inputs.input_transforms:
            cut_avt = in_transform.split('run-')[1]
            cut_apr = cut_avt.split('_')[0]
            run_nb_transforms.append(int(cut_apr))

        for order in self.inputs.stacks_order:
            index_img = run_nb_images.index(order)
            index_mask = run_nb_masks.index(order)
            index_tranform = run_nb_transforms.index(order)

            cmd += ['-i', self.inputs.input_images[index_img]]
            cmd += ['-m', self.inputs.input_masks[index_mask]]
            cmd += ['-t', self.inputs.input_transforms[index_tranform]]

        _, _, ext = split_filename(os.path.abspath(self.inputs.input_sdi))
        out_file = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                ''.join(([self.inputs.out_prefix, self.inputs.sub_ses, '_',
                                          str(len(self.inputs.stacks_order)),'V_rad',
                                          str(int(self.inputs.input_rad_dilatation)), ext])))

        cmd += ['-r', self.inputs.input_sdi]
        cmd += ['-o', out_file]

        if self.inputs.deblurring:
            cmd += ['--debluring']

        cmd += ['--loop', str(self.inputs.in_loop)]
        cmd += ['--deltat', str(self.inputs.in_deltat)]
        cmd += ['--lambda', str(self.inputs.in_lambda)]

        cmd += ['--bregman-loop', str(self.inputs.in_bregman_loop)]
        cmd += ['--iter', str(self.inputs.in_iter)]
        cmd += ['--step-scale', str(self.inputs.in_step_scale)]
        cmd += ['--gamma', str(self.inputs.in_gamma)]
        cmd += ['--inner-thresh', str(self.inputs.in_inner_thresh)]
        cmd += ['--outer-thresh', str(self.inputs.in_outer_thresh)]

        try:
            print('... cmd: {}'.format(cmd))
            cmd = ' '.join(cmd)
            run(self, cmd, env={}, cwd=os.path.abspath(self.inputs.bids_dir))
        except Exception as e:
            print('Failed')
            print(e)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        _, _, ext = split_filename(os.path.abspath(self.inputs.input_sdi))
        outputs['output_sr'] = os.path.join(os.getcwd().replace(self.inputs.bids_dir, '/fetaldata'),
                                            ''.join(([self.inputs.out_prefix, self.inputs.sub_ses, '_',
                                                      str(len(self.inputs.stacks_order)),'V_rad',
                                                      str(int(self.inputs.input_rad_dilatation)), ext])))

        return outputs
