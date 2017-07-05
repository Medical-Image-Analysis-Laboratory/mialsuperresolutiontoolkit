![MIALSRTK logo](https://cloud.githubusercontent.com/assets/22279770/24004342/5e78836a-0a66-11e7-8b7d-058961cfe8e8.png)
---

Copyright © 2016-2017 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland 

This software is distributed under the open-source BSD 3-Clause License. See LICENSE file for details.

---
The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) consists of a set of C++ image processing tools necessary to perform motion-robust super-resolution fetal MRI reconstruction. This toolkit, supported by the Swiss National Science Foundation (grant SNSF-141283), includes all algorithms and methods for brain extraction [1], intensity standardization [1,2], motion estimation and super-resolution [2] developed during my PhD. It uses the CMake build system and depends on the open-source image processing Insight ToolKit (ITK) library, the command line parser TCLAP library and OpenMP for multi-threading. The USAGE message of each tool can be obained using either the *-h* or *--help* flag. 
A Docker image is provided to facilitate their deployment.  

* Please acknowledge this software in any work reporting results using MIALSRTK by citing the following articles:

[1] S. Tourbier, C. Velasco-Annis, V. Taimouri, P. Hagmann, R. Meuli, S. K. Warfield, M. B. Cuadra,
A. Gholipour, *Automated template-based brain localization and extraction for fetal brain MRI
reconstruction*, Neuroimage (2017) In Press. [DOI](https://doi.org/10.1016/j.neuroimage.2017.04.004)

[2] S. Tourbier, X. Bresson, P. Hagmann, R. Meuli, M. B. Cuadra, *An efficient total variation
algorithm for super-resolution in fetal brain MRI with adaptive regularization*, Neuroimage 118
(2015) 584-597. [DOI](https://doi.org/10.1016/j.neuroimage.2015.06.018)

# Documentation #

* FOR USERS: [How to run Docker image](documentation/userguide_docker.md)
* FOR DEVELOPERS/CONTRIBUTORS: [Installation instructions on Ubuntu](documentation/devguide_ubuntu.md) / [Installation instructions on MACOSX](documentation/devguide_mac.md)
* [Doxygen source code documentation](https://htmlpreview.github.io/?https://github.com/sebastientourbier/mialsuperresolutiontoolkit/blob/master/documentation/doxygen_html/index.html)

# Credits #

* Sébastien Tourbier (sebastientourbier)
