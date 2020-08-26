# Copyright © 2016-2019 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

""" PyMIALSRTK utils functions
"""

import os

def run(self, command, env={}, cwd=os.getcwd()):
	import subprocess
	merged_env = os.environ
	merged_env.update(env)
	# Python 3.7
	# process = subprocess.run(command, shell=True,
	# 	env=merged_env, cwd=cwd, capture_output=True)

	# Python 3.6 (No capture_output)
	process = subprocess.run(command, shell=True,
		env=merged_env, cwd=cwd)
	return process
