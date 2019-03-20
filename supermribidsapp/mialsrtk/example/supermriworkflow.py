#!/usr/bin/env python

"""
Simple demo of a nipype pipeline that uses DataGrabber and DataSink.
"""

import os.path

import nipype
from mialsrtk.interfaces.preprocess import *
from mialsrtk.interfaces.reconstruction import *
from mialsrtk.interfaces.utils import *

# NLMDenoising, CorrectSliceIntensity,
# SliceBySliceN4BiasFieldCorrection, SliceBySliceCorrectBiasField, HistogramNormalization,
# IntensityStandardization, N4BiasFieldCorrection
#
# ImageReconstruction
# TVSuperResolution
#
# MaskImage
# OrientImage
# RefineHRMaskByIntersection

# Assume that inputs and outputs live in subdirectories of this directory:
base_dir = os.path.join('/home/tourbier/Desktop/VboxShare','ds-supermri')
subject = 'sub-eyesVD46a'

# Source and sink:
grabber = nipype.Node(interface=nipype.DataGrabber(infields=['arg'],
                                                   outfields=['out_file']),
                      name='grabber')
grabber.inputs.base_directory = os.path.join(base_dir, subject, 'anat')
grabber.inputs.sort_filelist = False
grabber.inputs.template = '%s.nii'
grabber.inputs.arg = '*'

# Use substition to force all output files to be dumped into the same
# directory:
sink = nipype.Node(interface=nipype.DataSink(),
                   name='sink')
sink.inputs.base_directory = os.path.join(base_dir, 'output')
sink.inputs.regexp_substitutions = [('_\w+\d+', '.')]

# Brain extraction:
bet = nipype.MapNode(interface=fsl.BET(), name='bet', iterfield=['in_file'])
bet.inputs.frac = 0.4
bet.inputs.reduce_bias = True

# Intensity normalization:
fslmaths = nipype.MapNode(interface=fsl.ImageMaths(op_string='-inm 0.5'),
                          name='fslmaths', iterfield=['in_file'])

# Use @ to prevent creation of subdirectories in sink's base directory
# when saving output:
workflow = nipype.Workflow('workflow')
workflow.connect([(grabber, bet, [('out_file', 'in_file')]),
                  (bet, fslmaths, [('out_file', 'in_file')]),
                  (fslmaths, sink, [('out_file', '@in_file.@final')])])

workflow.run('MultiProc', plugin_args={'n_procs': 4})
