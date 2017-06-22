![MIALSRTK logo](https://cloud.githubusercontent.com/assets/22279770/24004342/5e78836a-0a66-11e7-8b7d-058961cfe8e8.png)

# Instructions for use of installation-free Docker image #
# Tested on Ubuntu 16.04 #

## Docker installation
Install Docker, with apt-get

```
#!bash

sudo apt-get install docker docker-compose docker-registry docker.io python-docker python-dockerpty

```
## To run lastest version of MIALSRTK docker  image 

1) Make sure no docker mialsuperresolutiontoolkit container are already running

```
#!bash

sudo docker ps

```
2) If a container is already running, identify its iD (CID) from the previous command results and remove it

```
#!bash

sudo docker rm -f CID

```
3) Make sure no other docker image of mialsrtk are stored

```
#!bash

sudo docker images

```
4) If an image is already existing, identify from the previous command results the image iD (IID) and remove it

```
#!bash

sudo docker rmi -f IID

```
5) Retrieve the MIALSRTK docker image at its latest vesion

```
#!bash

sudo docker pull sebastientourbier/mialsuperresolutiontoolkit

```
6) Run the docker image

```
#!bash

sudo docker run -it sebastientourbier/mialsuperresolutiontoolkit

```
## For testing

1) Go to data folder

```
#!bash

cd ../data/

```
2) Run super-resolution pipeline

```
#!bash

sh superresolution_autoloc.sh listScansRECONauto.txt

```
## For your own data

1) Prepare your local volume to be mounted on docker

2) Run the docker image with local volume (/home/localadmin/Desktop/Jakab) mounted (as /fetaldata) 

```
#!bash

sudo docker run -v /home/localadmin/Desktop/Jakab:/fetaldata -it sebastientourbier/mialsuperresolutiontoolkit

```
3) Go to mounted volume

```
#!bash

cd /fetaldata

```
4) Run super-resolution pipeline

```
#!bash
sh superresolution_autoloc.sh listScansRECONauto.txt

```
# For developers/contributors

# Installation of MIALSRTK and dependencies on Ubuntu 16.04 LTS 

## Dependencies (Compilers: gcc/g++ 4.8 / ITK 4.9 / VTK 6.2 / CMake-gui 3.5.1/ TCLAP 1.2.1 / ANN 1.1.2)

```
#!bash

sudo apt-get install essential... (compilers...)
sudo apt-get install cmake-gui
sudo apt-get install libvtk5-dev libvtk5-qt4-dev 
sudo apt-get install libinsighttoolkit4.9 libinsighttoolkit4-dev
sudo apt-get install libtclap-dev
sudo apt-get install libncurses5 libncurses5-dev
sudo apt-get install ann-tools libann-dev
sudo apt-get install libgdcm2-dev libvtkgdcm2-dev libgdcm-tools libvtkgdcm-tools

```
## MIALSRTK ##

1) Go to your installation dir (YOUR_DIR):

```
#!bash

cd YOUR_DIR

```
2) Clone MIALSRTK from the github repository:

```
#!bash

git clone git@github.com:sebastientourbier/mialsuperresolutiontoolkit.git

```
3) Go into MIALSRTK local git repo:

```
#!bash

cd mialsuperresolutiontoolkit

```
4) Create a build folder

```
#!bash

mkdir build

```
5) Configure MIALSRTK using CMake gui with the following flags:   

```
#!cmake

 ANN_INCLUDE_DIR		  /usr/include/ANN
 ANN_LIBRARY			  /usr/lib/libann.so                 
 BUILD_DOCUMENTATION              OFF                                                         
 CMAKE_AR                         /usr/bin/ar                                                 
 CMAKE_BUILD_TYPE                                                                             
 CMAKE_COLOR_MAKEFILE             ON                                                          
 CMAKE_CXX_COMPILER               /usr/bin/g++                                                
 CMAKE_CXX_FLAGS                                                                              
 CMAKE_CXX_FLAGS_DEBUG            -g                                                          
 CMAKE_CXX_FLAGS_MINSIZEREL       -Os -DNDEBUG                                                
 CMAKE_CXX_FLAGS_RELEASE          -O3 -DNDEBUG                                                
 CMAKE_CXX_FLAGS_RELWITHDEBINFO   -O2 -g -DNDEBUG                                             
 CMAKE_C_COMPILER                 /usr/bin/gcc                                                 
 CMAKE_C_FLAGS                                                                                
 CMAKE_C_FLAGS_DEBUG              -g                                                          
 CMAKE_C_FLAGS_MINSIZEREL         -Os -DNDEBUG                                                
 CMAKE_C_FLAGS_RELEASE            -O3 -DNDEBUG                                                
 CMAKE_C_FLAGS_RELWITHDEBINFO     -O2 -g -DNDEBUG                                             
 CMAKE_EXE_LINKER_FLAGS                                                                       
 CMAKE_EXE_LINKER_FLAGS_DEBUG                                                                 
 CMAKE_EXE_LINKER_FLAGS_MINSIZE                                                               
 CMAKE_EXE_LINKER_FLAGS_RELEASE                                                               
 CMAKE_EXE_LINKER_FLAGS_RELWITH   
 CMAKE_EXPORT_COMPILE_COMMANDS    OFF                                                         
 CMAKE_INSTALL_NAME_TOOL          /usr/bin/install_name_tool                                  
 CMAKE_INSTALL_PREFIX             /usr/local                                                  
 CMAKE_LINKER                     /usr/bin/ld                                                 
 CMAKE_MAKE_PROGRAM               /usr/bin/make                                               
 CMAKE_MODULE_LINKER_FLAGS                                                                    
 CMAKE_MODULE_LINKER_FLAGS_DEBU                                                               
 CMAKE_MODULE_LINKER_FLAGS_MINS                                                               
 CMAKE_MODULE_LINKER_FLAGS_RELE                                                               
 CMAKE_MODULE_LINKER_FLAGS_RELW                                                               
 CMAKE_NM                         /usr/bin/nm                                                 
 CMAKE_OBJCOPY                    CMAKE_OBJCOPY-NOTFOUND                                      
 CMAKE_OBJDUMP                    CMAKE_OBJDUMP-NOTFOUND                                      
 CMAKE_OSX_ARCHITECTURES                                                                      
 CMAKE_OSX_DEPLOYMENT_TARGET                                                                  
 CMAKE_OSX_SYSROOT                                                                            
 CMAKE_RANLIB                     /usr/bin/ranlib                                             
 CMAKE_SHARED_LINKER_FLAGS                                                                    
 CMAKE_SHARED_LINKER_FLAGS_DEBU                                                               
 CMAKE_SHARED_LINKER_FLAGS_MINS                                                               
 CMAKE_SHARED_LINKER_FLAGS_RELE                                                               
 CMAKE_SHARED_LINKER_FLAGS_RELW                                                               
 CMAKE_SKIP_INSTALL_RPATH         OFF              
 CMAKE_SKIP_RPATH                 OFF                                                         
 CMAKE_STATIC_LINKER_FLAGS                                                                    
 CMAKE_STATIC_LINKER_FLAGS_DEBU                                                               
 CMAKE_STATIC_LINKER_FLAGS_MINS                                                               
 CMAKE_STATIC_LINKER_FLAGS_RELE                                                               
 CMAKE_STATIC_LINKER_FLAGS_RELW                                                               
 CMAKE_STRIP                      /usr/bin/strip                                              
 CMAKE_USE_RELATIVE_PATHS         OFF                                                         
 CMAKE_VERBOSE_MAKEFILE           OFF                                                         
 GIT_EXECUTABLE			  /usr/bin/git
 ITK_DIR			  /usr/lib/cmake/ITK-4.9
 PYTHON				  /usr/bin/python
 TCLAP_DIRECTORY		  /usr/include/tclap
 USE_OMP			  ON
 VTK_DIR			  /usr/lib/vtk-5.10
```

6) Compile

```
#!bash

make -j8
```

##For contributors, git configuration (if not done) 
```
#!bash
git config --global user.name "Your git username"
git config --global user.email "Your email "
```
# Installation of MIALSRTK and dependencies on MAC OSX 10.9.2 #

## System prerequisites ##

1) Install homebrew:

```
#!bash

ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"

```
2) Check home-brew environment:

```
#!bash

brew doctor
```
Follow the advices
3) Install cmake:

```
#!bash

brew install cmake

```
4) Install tclap:

```
#!bash

brew install tclap
```
5) Install ANN:

```
#!bash

brew install homebrew/science/ann

```
6) Install git:

```
#!bash

brew install git

```

7) Install doxygen:

```
#!bash

brew install doxygen

```
8) Install python:

```
#!bash

brew install python

```
9) Install ncurses (required by FetalInitializer)

```
#!bash

brew install ncurses

```
10) Add to .bashrc or .profile ulimit -s unlimited (If not done, sometimes FetalInitializer crashes)

## VTK ##

1) Install vtk 5.10.1 with Qt and python:

```
#!bash

brew install vtk5 —-with-qt —with-python

```
2) Configure/Edit environment variables in ~/.profile (if standard mac shell) or in ~/.zshrc:

```
#!bash

open -a TextEdit ~/.zshrc

```
Add the following line:
export VTK_DIR=/usr/local/Cellar/vtk5/5.10.1

## CMake gui (Not included in homebrew) ##

Download and install binaries (version cmake-2.8.12.2-Darwin64-universal) from [http://www.cmake.org/cmake/resources/software.html](Link URL)

## ITK ##

1) Download Archive of ITK 4.5.0 on Kitware website
2) Unarchive ITK
3) Configure ITK using CMake gui with the following flags:

```
#!cmake

 BUILD_DOCUMENTATION              OFF                                                         
 BUILD_EXAMPLES                   OFF                                                         
 BUILD_SHARED_LIBS                OFF                                                         
 BUILD_TESTING                    OFF                                                         
 BZRCOMMAND                       BZRCOMMAND-NOTFOUND                                         
 CMAKE_AR                         /usr/bin/ar                                                 
 CMAKE_BUILD_TYPE                 Release                                                     
 CMAKE_COLOR_MAKEFILE             ON                                                          
 CMAKE_CXX_COMPILER               /usr/bin/c++                                                
 CMAKE_CXX_FLAGS                                                                              
 CMAKE_CXX_FLAGS_DEBUG            -g                                                          
 CMAKE_CXX_FLAGS_MINSIZEREL       -Os -DNDEBUG                                                
 CMAKE_CXX_FLAGS_RELEASE          -O3 -DNDEBUG                                                
 CMAKE_CXX_FLAGS_RELWITHDEBINFO   -O2 -g -DNDEBUG                                             
 CMAKE_C_COMPILER                 /usr/bin/cc                                                 
 CMAKE_C_FLAGS                                                                                
 CMAKE_C_FLAGS_DEBUG              -g                                                          
 CMAKE_C_FLAGS_MINSIZEREL         -Os -DNDEBUG                                                
 CMAKE_C_FLAGS_RELEASE            -O3 -DNDEBUG                                                
 CMAKE_C_FLAGS_RELWITHDEBINFO     -O2 -g -DNDEBUG                                             
 CMAKE_EXE_LINKER_FLAGS                                                                       
 CMAKE_EXE_LINKER_FLAGS_DEBUG                                                                 
 CMAKE_EXE_LINKER_FLAGS_MINSIZE      
 CMAKE_EXE_LINKER_FLAGS_RELEASE                                                               
 CMAKE_EXE_LINKER_FLAGS_RELWITH                                                               
 CMAKE_EXPORT_COMPILE_COMMANDS    OFF                                                         
 CMAKE_INSTALL_NAME_TOOL          /usr/bin/install_name_tool                                  
 CMAKE_INSTALL_PREFIX             /usr/local                                                  
 CMAKE_LINKER                     /usr/bin/ld                                                 
 CMAKE_MAKE_PROGRAM               /usr/bin/make                                               
 CMAKE_MODULE_LINKER_FLAGS                                                                    
 CMAKE_MODULE_LINKER_FLAGS_DEBU                                                               
 CMAKE_MODULE_LINKER_FLAGS_MINS                                                               
 CMAKE_MODULE_LINKER_FLAGS_RELE                                                               
 CMAKE_MODULE_LINKER_FLAGS_RELW                                                               
 CMAKE_NM                         /usr/bin/nm                                                 
 CMAKE_OBJCOPY                    CMAKE_OBJCOPY-NOTFOUND                                      
 CMAKE_OBJDUMP                    CMAKE_OBJDUMP-NOTFOUND                                      
 CMAKE_OSX_ARCHITECTURES                                                                      
 CMAKE_OSX_DEPLOYMENT_TARGET                                                                  
 CMAKE_OSX_SYSROOT                                                                            
 CMAKE_RANLIB                     /usr/bin/ranlib                                             
 CMAKE_SHARED_LINKER_FLAGS                                                                    
 CMAKE_SHARED_LINKER_FLAGS_DEBU                                                               
 CMAKE_SHARED_LINKER_FLAGS_MINS                                                               
 CMAKE_SHARED_LINKER_FLAGS_RELE  
 CMAKE_SHARED_LINKER_FLAGS_RELW                                                               
 CMAKE_SKIP_INSTALL_RPATH         OFF                                                         
 CMAKE_SKIP_RPATH                 OFF                                                         
 CMAKE_STATIC_LINKER_FLAGS                                                                    
 CMAKE_STATIC_LINKER_FLAGS_DEBU                                                               
 CMAKE_STATIC_LINKER_FLAGS_MINS                                                               
 CMAKE_STATIC_LINKER_FLAGS_RELE                                                               
 CMAKE_STATIC_LINKER_FLAGS_RELW                                                               
 CMAKE_STRIP                      /usr/bin/strip                                              
 CMAKE_THREAD_LIBS                                                                            
 CMAKE_USE_RELATIVE_PATHS         OFF                                                         
 CMAKE_VERBOSE_MAKEFILE           OFF                                                         
 COREFOUNDATION_LIBRARY           /System/Library/Frameworks/CoreFoundation.framework         
 COVERAGE_COMMAND                 /usr/bin/gcov                                               
 COVERAGE_EXTRA_FLAGS             -l                                                          
 CPACK_BINARY_BUNDLE              OFF                                                         
 CPACK_BINARY_DEB                 OFF                                                         
 CPACK_BINARY_DRAGNDROP           OFF                                                         
 CPACK_BINARY_NSIS                OFF                                                         
 CPACK_BINARY_OSXX11              OFF                                                         
 CPACK_BINARY_PACKAGEMAKER        ON                                                          
 CPACK_BINARY_RPM                 OFF                                                         
 CPACK_BINARY_STGZ                ON                                              
  CPACK_BINARY_TBZ2                OFF                                                         
 CPACK_BINARY_TGZ                 ON                                                          
 CPACK_SOURCE_TBZ2                ON                                                          
 CPACK_SOURCE_TGZ                 ON                                                          
 CPACK_SOURCE_TZ                  ON                                                          
 CPACK_SOURCE_ZIP                 OFF                                                         
 CPPCHECK_EXECUTABLE              CPPCHECK_EXECUTABLE-NOTFOUND                                
 CPPCHECK_ROOT_DIR                                                                            
 CTEST_SUBMIT_RETRY_COUNT         3                                                           
 CTEST_SUBMIT_RETRY_DELAY         5                                                           
 CVSCOMMAND                       CVSCOMMAND-NOTFOUND                                         
 CVS_UPDATE_OPTIONS               -d -A -P                                                    
 DART_TESTING_TIMEOUT             1500                                                        
 EXECINFO_LIB                     EXECINFO_LIB-NOTFOUND                                       
 ExternalData_OBJECT_STORES                                                                   
 ExternalData_URL_TEMPLATES                                                                   
 GITCOMMAND                       /usr/bin/git                                                
 GIT_EXECUTABLE                   /usr/bin/git                                                
 HDF5_BUILD_WITH_INSTALL_NAME     OFF                                                         
 HGCOMMAND                        HGCOMMAND-NOTFOUND                                          
 ITKV3_COMPATIBILITY              ON                                                          
 ITK_BUILD_DEFAULT_MODULES        ON                                                          
 ITK_COMPUTER_MEMORY_SIZE         1                   
 ITK_CPPCHECK_TEST                OFF                                                         
 ITK_DOXYGEN_CHM                  OFF                                                         
 ITK_DOXYGEN_DOCSET               OFF                                                         
 ITK_DOXYGEN_ECLIPSEHELP          OFF                                                         
 ITK_DOXYGEN_HTML                 ON                                                          
 ITK_DOXYGEN_LATEX                OFF                                                         
 ITK_DOXYGEN_QHP                  OFF                                                         
 ITK_DOXYGEN_RTF                  OFF                                                         
 ITK_DOXYGEN_XML                  OFF                                                         
 ITK_LEGACY_REMOVE                OFF                                                         
 ITK_LEGACY_SILENT                OFF                                                         
 ITK_USE_64BITS_IDS               OFF                                                         
 ITK_USE_BRAINWEB_DATA            OFF                                                         
 ITK_USE_CONCEPT_CHECKING         ON                                                          
 ITK_USE_FFTWD                    OFF                                                         
 ITK_USE_FFTWF                    OFF                                                         
 ITK_USE_FLOAT_SPACE_PRECISION    OFF                                                         
 ITK_USE_GPU                      OFF                                                         
 ITK_USE_KWSTYLE                  OFF                                                         
 ITK_USE_STRICT_CONCEPT_CHECKIN   OFF                                                         
 ITK_USE_SYSTEM_DOUBLECONVERSIO   OFF                                                         
 ITK_USE_SYSTEM_FFTW              OFF                                                         
 ITK_USE_SYSTEM_GDCM              OFF                          
 ITK_USE_SYSTEM_HDF5              OFF                                                         
 ITK_USE_SYSTEM_JPEG              OFF                                                         
 ITK_USE_SYSTEM_PNG               OFF                                                         
 ITK_USE_SYSTEM_TIFF              OFF                                                         
 ITK_USE_SYSTEM_VXL               OFF                                                         
 ITK_USE_SYSTEM_ZLIB              OFF                                                         
 ITK_WRAPPING                     OFF                                                         
 ITK_WRAP_PYTHON                  OFF                                                         
 KWSTYLE_EXECUTABLE               KWSTYLE_EXECUTABLE-NOTFOUND                                 
 LSTK_USE_VTK                     OFF                                                         
 MAKECOMMAND                      /usr/bin/make -i                                            
 MAXIMUM_NUMBER_OF_HEADERS        35                                                          
 MEMORYCHECK_COMMAND              MEMORYCHECK_COMMAND-NOTFOUND                                
 MEMORYCHECK_SUPPRESSIONS_FILE                                                                
 Module_ITKDCMTK                  OFF                                                         
 Module_ITKIODCMTK                OFF                                                         
 Module_ITKIOMINC                 OFF                                                         
 Module_ITKIOPhilipsREC           OFF                                                         
 Module_ITKLevelSetsv4Visualiza   ON                                                          
 Module_ITKMINC                   OFF                                                         
 Module_ITKReview                 ON                                                          
 Module_ITKVideoBridgeOpenCV      OFF                                                         
 Module_ITKVideoBridgeVXL         OFF              
 Module_LesionSizingToolkit       ON                                                          
 Module_MGHIO                     OFF                                                         
 Module_SCIFIO                    OFF                                                         
 Module_SmoothingRecursiveYvvGa   OFF                                                         
 PERL_EXECUTABLE                  /usr/bin/perl                                               
 PYTHON_EXECUTABLE                /usr/bin/python                                             
 SCPCOMMAND                       /usr/bin/scp                                                
 SITE                             Sebastiens-MacBook-Pro.local                                
 SLURM_SBATCH_COMMAND             SLURM_SBATCH_COMMAND-NOTFOUND                               
 SLURM_SRUN_COMMAND               SLURM_SRUN_COMMAND-NOTFOUND                                 
 SVNCOMMAND                       /usr/bin/svn                                                
 SZIP_USE_EXTERNAL                OFF                                                         
 USE_COMPILER_HIDDEN_VISIBILITY   ON                                                          
 VCL_INCLUDE_CXX_0X               OFF                                                         
 VNL_CONFIG_CHECK_BOUNDS          ON                                                          
 VNL_CONFIG_ENABLE_SSE2           OFF                                                         
 VNL_CONFIG_ENABLE_SSE2_ROUNDIN   ON                                                          
 VNL_CONFIG_LEGACY_METHODS        OFF                                                         
 VNL_CONFIG_THREAD_SAFE           ON                                                          
 VTK_DIR                          /usr/local/Cellar/vtk5/5.10.1/lib/vtk-5.10                  
 VXL_UPDATE_CONFIGURATION         OFF                                                         
 ZLIB_USE_EXTERNAL                OFF  
```     
4) Compile

```
#!bash

make -j8
```
5) Install (run under admin privileges may be required)

```
#!bash

(sudo) make install
```
## MIALSRTK ##

1) Go to your installation dir (YOUR_DIR):

```
#!bash

cd YOUR_DIR

```
2) Clone MIALSRTK from the github repository:

```
#!bash

git clone git@github.com:sebastientourbier/mialsuperresolutiontoolkit.git

```
3) Go into MIALSRTK local git repo:

```
#!bash

cd mialsuperresolutiontoolkit

```
4) Create a build folder

```
#!bash

mkdir build

```
5) Configure MIALSRTK using CMake gui with the following flags:          

```
#!cmake

 ANN_INCLUDE_DIR                  /usr/local/Cellar/ann/1.1.2/include/ANN                     
 ANN_LIBRARY                      /usr/local/Cellar/ann/1.1.2/lib/libANN.a                    
 BUILD_DOCUMENTATION              OFF                                                         
 CMAKE_AR                         /usr/bin/ar                                                 
 CMAKE_BUILD_TYPE                                                                             
 CMAKE_COLOR_MAKEFILE             ON                                                          
 CMAKE_CXX_COMPILER               /usr/bin/c++                                                
 CMAKE_CXX_FLAGS                                                                              
 CMAKE_CXX_FLAGS_DEBUG            -g                                                          
 CMAKE_CXX_FLAGS_MINSIZEREL       -Os -DNDEBUG                                                
 CMAKE_CXX_FLAGS_RELEASE          -O3 -DNDEBUG                                                
 CMAKE_CXX_FLAGS_RELWITHDEBINFO   -O2 -g -DNDEBUG                                             
 CMAKE_C_COMPILER                 /usr/bin/cc                                                 
 CMAKE_C_FLAGS                                                                                
 CMAKE_C_FLAGS_DEBUG              -g                                                          
 CMAKE_C_FLAGS_MINSIZEREL         -Os -DNDEBUG                                                
 CMAKE_C_FLAGS_RELEASE            -O3 -DNDEBUG                                                
 CMAKE_C_FLAGS_RELWITHDEBINFO     -O2 -g -DNDEBUG                                             
 CMAKE_EXE_LINKER_FLAGS                                                                       
 CMAKE_EXE_LINKER_FLAGS_DEBUG                                                                 
 CMAKE_EXE_LINKER_FLAGS_MINSIZE                                                               
 CMAKE_EXE_LINKER_FLAGS_RELEASE                                                               
 CMAKE_EXE_LINKER_FLAGS_RELWITH   
 CMAKE_EXPORT_COMPILE_COMMANDS    OFF                                                         
 CMAKE_INSTALL_NAME_TOOL          /usr/bin/install_name_tool                                  
 CMAKE_INSTALL_PREFIX             /usr/local                                                  
 CMAKE_LINKER                     /usr/bin/ld                                                 
 CMAKE_MAKE_PROGRAM               /usr/bin/make                                               
 CMAKE_MODULE_LINKER_FLAGS                                                                    
 CMAKE_MODULE_LINKER_FLAGS_DEBU                                                               
 CMAKE_MODULE_LINKER_FLAGS_MINS                                                               
 CMAKE_MODULE_LINKER_FLAGS_RELE                                                               
 CMAKE_MODULE_LINKER_FLAGS_RELW                                                               
 CMAKE_NM                         /usr/bin/nm                                                 
 CMAKE_OBJCOPY                    CMAKE_OBJCOPY-NOTFOUND                                      
 CMAKE_OBJDUMP                    CMAKE_OBJDUMP-NOTFOUND                                      
 CMAKE_OSX_ARCHITECTURES                                                                      
 CMAKE_OSX_DEPLOYMENT_TARGET                                                                  
 CMAKE_OSX_SYSROOT                                                                            
 CMAKE_RANLIB                     /usr/bin/ranlib                                             
 CMAKE_SHARED_LINKER_FLAGS                                                                    
 CMAKE_SHARED_LINKER_FLAGS_DEBU                                                               
 CMAKE_SHARED_LINKER_FLAGS_MINS                                                               
 CMAKE_SHARED_LINKER_FLAGS_RELE                                                               
 CMAKE_SHARED_LINKER_FLAGS_RELW                                                               
 CMAKE_SKIP_INSTALL_RPATH         OFF              
 CMAKE_SKIP_RPATH                 OFF                                                         
 CMAKE_STATIC_LINKER_FLAGS                                                                    
 CMAKE_STATIC_LINKER_FLAGS_DEBU                                                               
 CMAKE_STATIC_LINKER_FLAGS_MINS                                                               
 CMAKE_STATIC_LINKER_FLAGS_RELE                                                               
 CMAKE_STATIC_LINKER_FLAGS_RELW                                                               
 CMAKE_STRIP                      /usr/bin/strip                                              
 CMAKE_USE_RELATIVE_PATHS         OFF                                                         
 CMAKE_VERBOSE_MAKEFILE           OFF                                                         
 GIT_EXECUTABLE                   /usr/bin/git                                                
 ITK_DIR                          /usr/local/lib/cmake/ITK-4.5
 PYTHON                           /usr/bin/python
 TCLAP_DIRECTORY                  /usr/local/Cellar/tclap/1.2.1/include                       
 USE_OMP                          OFF                                                         
 USE_SYSTEM_BTK                   OFF
```
6) Compile

```
#!bash

make -j8
```
