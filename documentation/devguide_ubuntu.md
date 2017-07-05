![MIALSRTK logo](https://cloud.githubusercontent.com/assets/22279770/24004342/5e78836a-0a66-11e7-8b7d-058961cfe8e8.png)

Copyright © 2016-2017 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland 

---

# Installation of MIALSRTK and dependencies on Ubuntu 16.04 LTS # 

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
---


# Contact #

* Sébastien Tourbier - sebastien(dot)tourbier1(at)gmail(dot)com

---
