# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK reconstruction functions."""

import os
import json

from traits.api import *

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

import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from nilearn.plotting import plot_anat

from pymialsrtk.interfaces.utils import run, reorder_by_run_ids
from pymialsrtk.utils import EXEC_PATH

########################
# Image Reconstruction
########################


class MialsrtkImageReconstructionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkImageReconstruction interface."""

    in_roi = traits.Enum(
        "mask",
        "all",
        "box",
        "mask",
        desc="""Define region of interest (required):
                                   - `box`: Use intersections for roi calculation
                                   - `mask`: Use masks for roi calculation
                                   - `all`: Use the whole image FOV""",
        mandatory=True,
        usedefault=True,
    )
    input_masks = InputMultiPath(File(), desc="Masks of the input images")
    input_images = InputMultiPath(File(), desc="Input images")
    input_rad_dilatation = traits.Float(
        1.0,
        desc=(
            "Radius dilatation used in "
            "prior step to construct "
            "output filename"
        ),
        usedefault=True,
    )
    out_sdi_prefix = traits.Str(
        "SDI_",
        desc=(
            "Suffix added to construct output"
            "scattered data interpolation filename"
        ),
        usedefault=True,
    )
    out_transf_postfix = traits.Str(
        "_transform",
        desc=("Suffix added to construct output " "transformation filenames"),
        usedefault=True,
    )
    stacks_order = traits.List(
        mandatory=True,
        desc=("List of stack run-id that specify the " "order of the stacks"),
    )
    out_prefix = traits.Str(
        "SRTV_",
        desc="Prefix added to construct output super-resolution filename",
        usedefault=True,
    )
    sub_ses = traits.Str(
        "x",
        desc=(
            "Subject and session BIDS identifier to "
            "construct output filename"
        ),
        usedefault=True,
    )
    skip_svr = traits.Bool(
        default=False, desc=("Skip slice-to-volume registration.")
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkImageReconstructionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkImageReconstruction interface."""

    output_sdi = File(desc="Output scattered data interpolation image file")
    output_transforms = OutputMultiPath(
        File(), desc="Output transformation files"
    )


class MialsrtkImageReconstruction(BaseInterface):
    """Creates a high-resolution image from a set of low resolution images.

    It is built on the BTK implementation and implements the method, presented in [Rousseau2006]_,
    that iterates between slice-to-volume registration and reconstruction of a
    high-resolution image using scattered data interpolation. The method converges
    when the mean square error between the high-resolution images reconstructed in
    the last two iterations is lower than ``1e-6``.

    It is used to estimate slice motion prior to :class:`~pymialsrtk.reconstruction.MialsrtkTVSuperResolution`.

    References
    ------------
    .. [Rousseau2006] Rousseau et al.; Acad Radiol., 2006. `(link to paper) <https://doi.org/10.1016/j.acra.2006.05.003>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.reconstruction import MialsrtkImageReconstruction
    >>> srtkImageReconstruction = MialsrtkImageReconstruction()
    >>> srtkImageReconstruction.input_images = ['sub-01_ses-01_run-1_T2w.nii.gz',
    >>>                                         'sub-01_ses-01_run-2_T2w.nii.gz',
    >>>                                         'sub-01_ses-01_run-3_T2w.nii.gz',
    >>>                                         'sub-01_ses-01_run-4_T2w.nii.gz']
    >>> srtkImageReconstruction.input_masks = ['sub-01_ses-01_run-1_mask.nii.gz',
    >>>                                        'sub-01_ses-01_run-2_mask.nii.gz',
    >>>                                        'sub-01_ses-01_run-3_mask.nii.gz',
    >>>                                        'sub-01_ses-01_run-4_mask.nii.gz']
    >>> srtkImageReconstruction.inputs.stacks_order = [3,1,2,4]
    >>> srtkImageReconstruction.inputs.sub_ses = 'sub-01_ses-01'
    >>> srtkImageReconstruction.inputs.in_roi = 'mask'
    >>> srtkImageReconstruction.run()  # doctest: +SKIP

    """

    input_spec = MialsrtkImageReconstructionInputSpec
    output_spec = MialsrtkImageReconstructionOutputSpec

    def _gen_filename(self, orig, name):
        if name == "output_sdi":
            _, _, ext = split_filename(orig)
            output = "".join(
                [
                    self.inputs.out_sdi_prefix,
                    self.inputs.sub_ses,
                    "_",
                    str(len(self.inputs.stacks_order)),
                    "V_rad",
                    str(int(self.inputs.input_rad_dilatation)),
                    ext,
                ]
            )
            return os.path.abspath(output)

        elif name == "output_transforms":
            _, name, _ = split_filename(orig)
            output = "".join(
                [
                    name,
                    self.inputs.out_transf_postfix,
                    "_",
                    str(len(self.inputs.stacks_order)),
                    "V",
                    ".txt",
                ]
            )

            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        params = ["".join(["--", self.inputs.in_roi])]

        input_images = reorder_by_run_ids(
            self.inputs.input_images, self.inputs.stacks_order
        )
        input_masks = reorder_by_run_ids(
            self.inputs.input_masks, self.inputs.stacks_order
        )

        for in_image, in_mask in zip(input_images, input_masks):

            params.append("-i")
            params.append(in_image)

            if self.inputs.in_roi == "mask":
                params.append("-m")
                params.append(in_mask)

            transf_file = self._gen_filename(in_image, "output_transforms")
            params.append("-t")
            params.append(transf_file)

        out_file = self._gen_filename(
            self.inputs.input_images[0], "output_sdi"
        )
        params.append("-o")
        params.append(out_file)

        if self.inputs.skip_svr:
            params.append("--noreg")
        if self.inputs.verbose:
            params.append("--verbose")

        cmd = [f"{EXEC_PATH}mialsrtkImageReconstruction"]
        cmd += params

        if self.inputs.verbose:
            print("... cmd: {}".format(cmd))
        cmd = " ".join(cmd)
        run(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_transforms"] = [
            self._gen_filename(in_image, "output_transforms")
            for in_image in self.inputs.input_images
        ]
        outputs["output_sdi"] = self._gen_filename(
            self.inputs.input_images[0], "output_sdi"
        )
        return outputs


#####################################
#  Total Variation Super Resolution
#####################################


class MialsrtkTVSuperResolutionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the MialsrtkTVSuperResolution interface."""

    input_images = InputMultiPath(
        File(mandatory=True), desc="Input image filenames for super-resolution"
    )
    input_masks = InputMultiPath(
        File(mandatory=True), desc="Masks of input images for super-resolution"
    )
    input_transforms = InputMultiPath(
        File(mandatory=True),
        desc="Estimated slice-by-slice ITK transforms of input images",
    )
    input_sdi = File(
        desc="Reconstructed image for initialization. "
        "Typically the output of MialsrtkImageReconstruction is used",
        mandatory=True,
    )
    out_prefix = traits.Str(
        "SRTV_",
        desc="Prefix added to construct output super-resolution filename",
        usedefault=True,
    )
    stacks_order = traits.List(
        mandatory=False,
        desc="List of stack run-id that specify the " "order of the stacks",
    )
    in_loop = traits.Int(mandatory=True, desc="Number of loops (SR/denoising)")
    in_deltat = traits.Float(
        mandatory=True, desc="Parameter deltat of TV optimizer"
    )
    in_lambda = traits.Float(
        mandatory=True,
        desc="TV regularization factor which weights the data fidelity term in TV optimizer",
    )
    in_bregman_loop = traits.Int(
        3, desc="Number of Bregman loops", usedefault=True
    )
    in_iter = traits.Int(
        50, desc="Number of inner iterations", usedefault=True
    )
    in_step_scale = traits.Int(
        10, desc="Parameter step scale", usedefault=True
    )
    in_gamma = traits.Int(10, desc="Parameter gamma", usedefault=True)
    in_inner_thresh = traits.Float(
        0.00001, desc="Inner loop convergence threshold", usedefault=True
    )
    in_outer_thresh = traits.Float(
        0.000001, desc="Outer loop convergence threshold", usedefault=True
    )
    deblurring = traits.Bool(
        False,
        desc="Flag to set deblurring PSF during SR (double the neighborhood)",
        usedefault=True,
    )
    input_rad_dilatation = traits.Float(
        1.0,
        desc="Radius dilatation used in prior step to construct output filename",
        usedefault=True,
    )
    sub_ses = traits.Str(
        "x",
        desc="Subject and session BIDS identifier to construct output filename",
        usedefault=True,
    )
    use_manual_masks = traits.Bool(
        False, desc="Use masks of input files", usedefault=True
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkTVSuperResolutionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MialsrtkTVSuperResolution interface."""

    output_sr = File(desc="Output super-resolution image file")
    output_sr_png = File(
        desc="Output super-resolution PNG image file for quality assessment"
    )
    output_json_path = File(
        desc="Output json file where super-resolution reconstruction parameters "
        "are summarized"
    )
    output_TV_parameters = traits.Dict(
        desc="Dictionary of all TV parameters used"
    )


class MialsrtkTVSuperResolution(BaseInterface):
    """Apply super-resolution algorithm using one or multiple input images.

    It implements the super-resolution algorithm with Total-Variation regularization presented in [Tourbier2015]_.

    Taking as input the list of input low resolution images and the list of slice-by-slice
    transforms estimated with :class:`~pymialsrtk.reconstruction.MialsrtkImageReconstruction`,
    it builds the forward model `H` that relates the low resolution images `y` to the high resolution
    `x` to be estimated by `y = Hx`, and it solves optimally using convex optimization
    the inverse (i.e. super-resolution) problem (finding `x`) with exact Total-Variation regularization.

    The weight of TV regularization is controlled by `in_lambda`. The lower it is, the lower
    will be the weight of the data fidelity and so the higher will the regularization.
    The optimization time step is controlled by `in_deltat` that defines the time step of the optimizer.

    References
    ------------
    .. [Tourbier2015] Tourbier et al.; NeuroImage, 2015. `(link to paper) <https://doi.org/10.1016/j.neuroimage.2015.06.018>`_

    Example
    ----------
    >>> from pymialsrtk.interfaces.reconstruction import MialsrtkTVSuperResolution
    >>> srtkTVSuperResolution = MialsrtkTVSuperResolution()
    >>> srtkTVSuperResolution.input_images = ['sub-01_ses-01_run-1_T2w.nii.gz', 'sub-01_ses-01_run-2_T2w.nii.gz', \
    'sub-01_ses-01_run-3_T2w.nii.gz', 'sub-01_ses-01_run-4_T2w.nii.gz']
    >>> srtkTVSuperResolution.input_masks = ['sub-01_ses-01_run-1_mask.nii.gz', 'sub-01_ses-01_run-2_mask.nii.gz', \
    'sub-01_ses-01_run-3_mask.nii.gz', 'sub-01_ses-01_run-4_mask.nii.gz']
    >>> srtkTVSuperResolution.input_transforms = ['sub-01_ses-01_run-1_transform.txt', 'sub-01_ses-01_run-2_transform.txt', \
    'sub-01_ses-01_run-3_transform.txt', 'sub-01_ses-01_run-4_transform.txt']
    >>> srtkTVSuperResolution.input_sdi = 'sdi.nii.gz'
    >>> srtkTVSuperResolution.inputs.stacks_order = [3,1,2,4]
    >>> srtkTVSuperResolution.inputs.sub_ses = 'sub-01_ses-01'
    >>> srtkTVSuperResolution.inputs.in_loop = 10
    >>> srtkTVSuperResolution.inputs.in_deltat = 0.01
    >>> srtkTVSuperResolution.inputs.in_lambda = 0.75
    >>> srtkTVSuperResolution.run()  # doctest: +SKIP

    """

    input_spec = MialsrtkTVSuperResolutionInputSpec
    output_spec = MialsrtkTVSuperResolutionOutputSpec

    m_out_files = ""
    m_output_dict = {}

    m_TV_parameters = {}

    def _gen_filename(self, name):
        if name == "output_sr":
            _, _, ext = split_filename(self.inputs.input_sdi)
            output = "".join(
                [
                    self.inputs.out_prefix,
                    self.inputs.sub_ses,
                    "_",
                    str(len(self.inputs.stacks_order)),
                    "V_rad",
                    str(int(self.inputs.input_rad_dilatation)),
                    ext,
                ]
            )
            return os.path.abspath(output)

        elif name == "output_json_path":
            output = "".join(
                [
                    self.inputs.out_prefix,
                    self.inputs.sub_ses,
                    "_",
                    str(len(self.inputs.stacks_order)),
                    "V_rad",
                    str(int(self.inputs.input_rad_dilatation)),
                    ".json",
                ]
            )

            return os.path.abspath(output)

        elif name == "output_sr_png":
            output = "".join(
                [
                    self.inputs.out_prefix,
                    self.inputs.sub_ses,
                    "_",
                    str(len(self.inputs.stacks_order)),
                    "V_rad",
                    str(int(self.inputs.input_rad_dilatation)),
                    ".png",
                ]
            )

            return os.path.abspath(output)

        return None

    def _run_interface(self, runtime):

        cmd = [f"{EXEC_PATH}mialsrtkTVSuperResolution"]

        input_images = reorder_by_run_ids(
            self.inputs.input_images, self.inputs.stacks_order
        )
        input_masks = reorder_by_run_ids(
            self.inputs.input_masks, self.inputs.stacks_order
        )
        input_transforms = reorder_by_run_ids(
            self.inputs.input_transforms, self.inputs.stacks_order
        )

        for in_image, in_mask, in_transform in zip(
            input_images, input_masks, input_transforms
        ):
            cmd += ["-i", in_image]
            cmd += ["-m", in_mask]
            cmd += ["-t", in_transform]

        cmd += ["-r", self.inputs.input_sdi]

        out_sr = self._gen_filename("output_sr")
        cmd += ["-o", out_sr]

        if self.inputs.deblurring:
            cmd += ["--debluring"]

        self.m_TV_parameters["in_loop"] = str(self.inputs.in_loop)
        self.m_TV_parameters["in_deltat"] = str(self.inputs.in_deltat)
        self.m_TV_parameters["in_lambda"] = str(self.inputs.in_lambda)
        self.m_TV_parameters["in_bregman_loop"] = str(
            self.inputs.in_bregman_loop
        )
        self.m_TV_parameters["in_iter"] = str(self.inputs.in_iter)
        self.m_TV_parameters["in_step_scale"] = str(self.inputs.in_step_scale)
        self.m_TV_parameters["in_gamma"] = str(self.inputs.in_gamma)
        self.m_TV_parameters["in_inner_thresh"] = str(
            self.inputs.in_inner_thresh
        )
        self.m_TV_parameters["in_outer_thresh"] = str(
            self.inputs.in_outer_thresh
        )

        cmd += ["--loop", str(self.inputs.in_loop)]
        cmd += ["--deltat", str(self.inputs.in_deltat)]
        cmd += ["--lambda", str(self.inputs.in_lambda)]
        cmd += ["--bregman-loop", str(self.inputs.in_bregman_loop)]
        cmd += ["--iter", str(self.inputs.in_iter)]
        cmd += ["--step-scale", str(self.inputs.in_step_scale)]
        cmd += ["--gamma", str(self.inputs.in_gamma)]
        cmd += ["--inner-thresh", str(self.inputs.in_inner_thresh)]
        cmd += ["--outer-thresh", str(self.inputs.in_outer_thresh)]
        if self.inputs.verbose:
            cmd += ["--verbose"]
        cmd = " ".join(cmd)
        run(cmd)

        ################################################
        # Creation of JSON sidecar file of the SR image
        ################################################
        self.m_output_dict["Description"] = (
            "Isotropic high-resolution image reconstructed using the "
            "Total-Variation Super-Resolution algorithm provided by MIALSRTK"
        )
        self.m_output_dict[
            "Input sources run order"
        ] = self.inputs.stacks_order
        self.m_output_dict["CustomMetaData"] = {}
        self.m_output_dict["CustomMetaData"]["n_stacks"] = str(
            len(self.inputs.stacks_order)
        )
        self.m_output_dict["CustomMetaData"]["masks_used"] = (
            "Manual" if self.inputs.use_manual_masks else "Automatic"
        )
        self.m_output_dict["CustomMetaData"][
            "in_deltat"
        ] = self.inputs.in_deltat
        self.m_output_dict["CustomMetaData"][
            "in_lambda"
        ] = self.inputs.in_lambda
        self.m_output_dict["CustomMetaData"]["in_loop"] = self.inputs.in_loop
        self.m_output_dict["CustomMetaData"][
            "in_bregman_loop"
        ] = self.inputs.in_bregman_loop
        self.m_output_dict["CustomMetaData"]["in_iter"] = self.inputs.in_iter
        self.m_output_dict["CustomMetaData"][
            "in_step_scale"
        ] = self.inputs.in_step_scale
        self.m_output_dict["CustomMetaData"]["in_gamma"] = self.inputs.in_gamma

        output_json_path = self._gen_filename("output_json_path")
        with open(output_json_path, "w") as outfile:
            if self.inputs.verbose:
                print("  > Write JSON side-car...")
            json.dump(self.m_output_dict, outfile, indent=4)

        #########################################################
        # Save cuts of the SR image in a PNG for later reporting
        #########################################################
        out_sr_png = self._gen_filename("output_sr_png")

        # Load the super-resolution image
        if self.inputs.verbose:
            print(f"  > Load SR image {out_sr}...")
        img = nib.load(out_sr)
        # Get image properties
        zooms = img.header.get_zooms()  # zooms are the size of the voxels
        shapes = img.shape
        fovs = np.array(zooms) * np.array(shapes)
        # Get middle cut
        cuts = [s // 2 for s in shapes]

        if self.inputs.verbose:
            print(
                f"    Image properties: Zooms={zooms}/ Shape={shapes}/ "
                f"FOV={fovs}/ middle cut={cuts}"
            )
        # Crop the image if the FOV exceeds a certain value

        def compute_axis_crop_indices(cut, fov, max_fov=120):
            """Compute the cropping index in a dimension if the Field-Of-View exceeds a maximum value of 120mm by default.

            Parameters
            ----------
            cut: int
                Middle slice index in a given dimension

            fov: float
                Slice Field-of-View (mm) in a given dimension

            max_fov: float
                Maximum Slice Field-of-View (mm) to which the image does not need to be cropped
                (120mm by default)

            Returns
            -------
            (crop_start_index, crop_end_index): (int, int)
                Starting and ending indices of the image crop along the given dimension

            """
            crop_start_index = cut - max_fov // 2 if fov > max_fov else 0
            crop_end_index = cut + max_fov // 2 if fov > max_fov else -1
            return crop_start_index, crop_end_index

        crop_start_x, crop_end_x = compute_axis_crop_indices(
            cuts[0], fovs[0], max_fov=120
        )
        crop_start_y, crop_end_y = compute_axis_crop_indices(
            cuts[1], fovs[1], max_fov=120
        )
        crop_start_z, crop_end_z = compute_axis_crop_indices(
            cuts[2], fovs[2], max_fov=120
        )
        if self.inputs.verbose:
            print(
                f"  > Crop SR image at "
                f"({crop_start_x}:{crop_end_x}, "
                f"{crop_start_y}:{crop_end_y}, "
                f"{crop_start_z}:{crop_end_z})..."
            )
        cropped_img = img.slicer[
            crop_start_x:crop_end_x,
            crop_start_y:crop_end_y,
            crop_start_z:crop_end_z,
        ]
        del img

        # Create and save the figure
        plot_anat(
            anat_img=cropped_img,
            annotate=True,
            draw_cross=False,
            black_bg=True,
            dim="auto",
            display_mode="ortho",
        )
        if self.inputs.verbose:
            print(f"Save the PNG image cuts as {out_sr_png}")
        plt.savefig(out_sr_png, dpi=100, facecolor="k", edgecolor="k")

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_sr"] = self._gen_filename("output_sr")
        outputs["output_sr_png"] = self._gen_filename("output_sr_png")
        outputs["output_json_path"] = self._gen_filename("output_json_path")
        outputs["output_TV_parameters"] = self.m_TV_parameters
        return outputs


########################
# Image Reconstruction
########################


class MialsrtkSDIComputationInputSpec(BaseInterfaceInputSpec):
    """Class used to represent inputs of the
    MialsrtkSDIComputation interface."""

    input_images = InputMultiPath(File(), desc="Input images", mandatory=True)
    input_masks = InputMultiPath(
        File(), desc="Masks of the input images", mandatory=True
    )
    input_transforms = InputMultiPath(
        File(), desc="Input transformation files", mandatory=True
    )
    input_reference = File(desc="Input reference image", mandatory=True)
    stacks_order = traits.List(
        mandatory=True,
        desc="List of stack run-id that " "specify the order of the stacks",
    )
    label_id = traits.Int(-1, mandatory=False, usedefault=True)
    sub_ses = traits.Str(
        "x",
        desc=(
            "Subject and session BIDS identifier to "
            "construct output filename"
        ),
        usedefault=True,
    )
    verbose = traits.Bool(desc="Enable verbosity")


class MialsrtkSDIComputationOutputSpec(TraitedSpec):
    """Class used to represent outputs of the
    MialsrtkSDIComputation interface."""

    output_sdi = File(desc="Output scattered data interpolation image file")


class MialsrtkSDIComputation(BaseInterface):
    """Creates a high-resolution image
    from a set of low resolution images and
    their slice-by-slicemotion parameters.
    """

    input_spec = MialsrtkSDIComputationInputSpec
    output_spec = MialsrtkSDIComputationOutputSpec

    def _gen_filename(self, name):
        if name == "output_sdi":
            if self.inputs.label_id != -1:
                output = "".join(
                    [
                        self.inputs.sub_ses,
                        "_",
                        "HR",
                        "_",
                        "label-" + str(self.inputs.label_id),
                        ".nii.gz",
                    ]
                )
            else:
                output = "".join([self.inputs.sub_ses, "_", "HR", ".nii.gz"])
            return os.path.abspath(output)

        return None

    def get_empty_ref_image(self, input_ref):
        """Generate an empty reference image for
        _run_interface
        """
        import SimpleITK as sitk

        reader = sitk.ImageFileReader()
        writer = sitk.ImageFileWriter()

        reader.SetFileName(input_ref)

        image = reader.Execute()

        multiply = sitk.MultiplyImageFilter()
        image = multiply.Execute(0, image)

        out_path = os.path.abspath(os.path.basename(input_ref))
        writer.SetFileName(out_path)
        writer.Execute(image)

        return out_path

    def _run_interface(self, runtime):

        # Reference image must be empty
        empty_ref_image = self.get_empty_ref_image(self.inputs.input_reference)
        params = []
        # params.append(''.join(["--", self.inputs.in_roi]))

        input_images = self.inputs.input_images

        if self.inputs.label_id != -1:
            input_images = [
                im
                for im in input_images
                if "label-" + str(self.inputs.label_id) + ".nii.gz" in im
            ]

        input_images = reorder_by_run_ids(
            input_images, self.inputs.stacks_order
        )
        input_masks = reorder_by_run_ids(
            self.inputs.input_masks, self.inputs.stacks_order
        )
        input_transforms = reorder_by_run_ids(
            self.inputs.input_transforms, self.inputs.stacks_order
        )

        params.append("-r")
        params.append(empty_ref_image)

        for in_image, in_mask, in_transform in zip(
            input_images, input_masks, input_transforms
        ):

            params.append("-i")
            params.append(in_image)

            params.append("-m")
            params.append(in_mask)

            params.append("-t")
            params.append(in_transform)

        out_file = self._gen_filename("output_sdi")
        params.append("-o")
        params.append(out_file)

        cmd = [f"{EXEC_PATH}mialsrtkSDIComputation"]
        cmd += params
        if self.inputs.verbose:
            cmd += ["--verbose"]
            print("... cmd: {}".format(cmd))
        cmd = " ".join(cmd)
        run(cmd)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["output_sdi"] = self._gen_filename("output_sdi")
        return outputs
