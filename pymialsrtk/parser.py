# Copyright © 2016-2019 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

""" MIALSRTK BIDS App Commandline Parser
"""

from info import __version__
from info import __release_date__

def get_parser():
    import argparse
    p = argparse.ArgumentParser(description='Entrypoint script to the MIALsrtk pipeline')
    p.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    p.add_argument('output_dir', help='The directory where the output files '
                        'should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')
    p.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                        'Only participant is available',
                        choices=['participant'])
    p.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                       'corresponds to sub-<participant_label> from the BIDS spec '
                       '(so it does not include "sub-"). If this parameter is not '
                       'provided all subjects should be analyzed. Multiple '
                       'participants can be specified with a space separated list.',
                       nargs="+")
    
    p.add_argument('--param_file', help='Path to a JSON file containing subjects\' exams ' 
                       'information and super-resolution total variation parameters.', 
                       default='/bids_dir/code/participants_param.json', type=str)
    p.add_argument('-v', '--version', action='version',
                        version='BIDS-App MIALSRTK version {}'.format(__version__))
    return p