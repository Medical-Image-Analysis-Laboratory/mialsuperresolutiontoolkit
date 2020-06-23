
MIAL Super-Resolution Toolkit
============================================

.. image:: images/mialsrtk-logo.png
  :width: 1000
  :align: center

**Latest released version:** |release|

This neuroimaging processing pipeline software is developed by the Medical Image Analysis Laboratory (MIAL) at the University Hospital of Lausanne (CHUV) for use within the lab, as well as for open-source software distribution.


.. image:: https://zenodo.org/badge/183162514.svg
  :target: https://zenodo.org/badge/latestdoi/183162514
  :alt: Digital Object Identifier
.. image:: https://img.shields.io/docker/pulls/sebastientourbier/mialsuperresolutiontoolkit
  :target: 
  :alt: Docker Pulls
.. image:: https://travis-ci.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit.svg?branch=master
  :target: https://travis-ci.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit
  :alt: TravisCI Status (C++)
.. image:: https://circleci.com/gh/connectomicslab/connectomemapper3/tree/master.svg?style=shield
  :target: https://circleci.com/gh/connectomicslab/connectomemapper3/tree/master
  :alt: CircleCI Status (BIDS-App)
.. image:: https://readthedocs.org/projects/connectome-mapper-3/badge/?version=latest
  :target: https://connectome-mapper-3.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://img.shields.io/github/all-contributors/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit
  :target:
  :alt: Github All Contributors  


.. warning:: THIS SOFTWARE IS FOR RESEARCH PURPOSES ONLY AND SHALL NOT BE USED FOR ANY CLINICAL USE. THIS SOFTWARE HAS NOT BEEN REVIEWED OR APPROVED BY THE FOOD AND DRUG ADMINISTRATION OR EQUIVALENT AUTHORITY, AND IS FOR NON-CLINICAL, IRB-APPROVED RESEARCH USE ONLY. IN NO EVENT SHALL DATA OR IMAGES GENERATED THROUGH THE USE OF THE SOFTWARE BE USED IN THE PROVISION OF PATIENT CARE.
  
  
*********
About
*********

The Medical Image Analysis Laboratory Super-Resolution ToolKit (MIALSRTK) [ref1]_ consists of a set of C++ and Python image processing tools necessary to perform motion-robust super-resolution fetal MRI reconstruction. 

The C++ MIALSRTK library includes all algorithms and methods for brain extraction [ref2]_ , intensity standardization [ref2]_ [ref3]_ , motion estimation and super-resolution [ref3]_  developed during the PhD of Sebastien Tourbier. It uses the CMake build system and depends on the open-source image processing Insight ToolKit (ITK) library, the command line parser TCLAP library and OpenMP for multi-threading. The USAGE message of each tool can be obtained using either the *-h* or *--help* flag. 

Adopting recent advances in standardization of neuroimaging data organization (with the Brain Imaging Data Structure - BIDS - standard) and processing (with the BIDS App standard), 
MIALSRTK2 provides now a BIDS App, a . Its workflow relies on the Nipype dataflow library which:
* represents the entire processing pipeline as a graph, where each MIALSRTK C++ tools are connected, and
* provides a mecanism to record data provenance and execution details. 

See :ref:`BIDS App usage <cmdusage>` for more details.  


*********
Funding
*********

Originally supported by the Swiss National Science Foundation (grant SNSF-141283).

*******************
License information
*******************

This software is distributed under the open-source license Modified BSD. See :ref:`license <LICENSE>` for more details.

All trademarks referenced herein are property of their respective holders.

*******************
Aknowledgment
*******************

If your are using the MIALSRTK BIDS App in your work, please acknowledge this software and its dependencies. See :ref:`Citing <citing>` for more details.

Help/Questions
--------------

If you run into any problems or have any code bugs or questions, please create a new `GitHub Issue <https://github.com/Medical-Image-Analysis-Laboratory/mialsuperresolutiontoolkit/issues>`_.

.. ***********************
.. Eager to contribute?
.. ***********************

.. See :ref:`Contributing to Connectome Mapper <contributing>` for more details.

***********************
Contents
***********************

.. _getting_started:

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation

.. _user-docs:

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   bids
   usage

.. _user-usecases:

.. toctree::
   :maxdepth: 1
   :caption: Examples & Tutorials
   
   NLM denoising <notebooks/brainHack.ipynb>

.. _about-docs:

.. toctree::
   :maxdepth: 1
   :caption: About MIALSRTK

   LICENSE
   citing
