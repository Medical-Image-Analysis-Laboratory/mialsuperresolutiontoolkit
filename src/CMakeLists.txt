PROJECT(MIALSRTK)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#TO BE FIXED: Setup environment variable
#SET(cmd "${CMAKE_SOURCE_DIR}/Scripts/envsetup.sh")

#SET(${MIALTK_DIR} "${CMAKE_BINARY_DIR}")

#MESSAGE(STATUS MIALTK_DIR=New3)

#ADD_CUSTOM_COMMAND(TARGET envsetup2 PRE_BUILD
#COMMAND  "sh" ARGS ${cmd} "Hello42"
#DEPENDS MIALTK
#COMMENT "Setting environment variables..."
#)

#ADD_CUSTOM_COMMAND(TARGET envsetup PRE_BUILD
#COMMAND  "python" ARGS "${CMAKE_SOURCE_DIR}/Scripts/envsetup.py" "Hello42"
#DEPENDS MIALTK
#COMMENT "Setting environment variables..."
#)

#EXECUTE_PROCESS(COMMAND /usr/bin/sh -c "export MIALTK_DIR=$ENV{MIALTK_DIR}")

#---- Options --------------------------------------------------------------------------

# Parallel computing
OPTION(USE_OMP "Use OpenMP" OFF)
IF(USE_OMP)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
ENDIF(USE_OMP)

# Warning mask
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-gnu -Wno-deprecated")

# Testing
#OPTION(BUILD_TESTING "Build testing applications" OFF)

# Doxygen documentation
INCLUDE(${CMAKE_ROOT}/Modules/Documentation.cmake)
OPTION(BUILD_DOCUMENTATION "Build the doxygen documentation" OFF)

# Python script support
#OPTION(BTK_ENABLE_SCRIPT "Enable the use of python script in BTK." OFF)

# Build Experimental programs
#OPTION(BTK_BUILD_EXPERIMENTAL "Enable the use of experimental programs in BTK " OFF)

#---- Dependencies ---------------------------------------------------------------------

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

# Find ITK
MESSAGE(STATUS "Check for ITK package")
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

# Find VTK package
MESSAGE(STATUS "Check for VTK package")
FIND_PACKAGE(VTK REQUIRED)
INCLUDE(${VTK_USE_FILE})

# Find ANN
FIND_PATH(ANN_INCLUDE_DIR ANN.h /usr/include/ANN)
IF(ANN_INCLUDE_DIR-NOTFOUND)
  MESSAGE(FATAL_ERROR "ANN header not found !")
ENDIF(ANN_INCLUDE_DIR-NOTFOUND)

SET(ANN_NAMES ${ANN_NAMES} ann)
FIND_LIBRARY(ANN_LIBRARY NAMES ${ANN_NAMES} )
IF(ANN_LIBRARY-NOTFOUND)
  MESSAGE(FATAL_ERROR "ANN library not found !")
ENDIF(ANN_LIBRARY-NOTFOUND)

# Find TCLAP
MESSAGE(STATUS "Check for TCLAP")
FIND_PATH(TCLAP_DIRECTORY CmdLine.h /usr/include/tclap)
IF(TCLAP_DIRECTORY-NOTFOUND)
  MESSAGE(FATAL_ERROR "TCLAP not found !")
ENDIF(TCLAP_DIRECTORY-NOTFOUND)

# Find Git
MESSAGE(STATUS "Check for GIT")
SET(GIT_NAMES git eg)
FIND_PROGRAM(GIT_EXECUTABLE 
  NAMES ${GIT_NAMES}
  DOC "Git command line client"
)



#---- Operations -----------------------------------------------------------------------


# Uninstall command
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(mialtk_uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

#set(BTK_INSTALL_DIR ${BTK_DIR}/../bin)
#set(BTK_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BTK_INSTALL_DIR})


# Executables
#add_subdirectory( External )

#External project BTK
#include(ExternalProject)

message(STATUS ITK_DIR=${ITK_DIR})
message(STATUS VTK_DIR=${VTK_DIR})
message(STATUS TCLAP_DIRECTORY=${TCLAP_DIRECTORY})
message(STATUS USE_OMP=${USE_OMP})

#message(STATUS BTK_INSTALL_DIR=${BTK_INSTALL_DIR})
#message(STATUS BTK_CMAKE_ARGS=${BTK_CMAKE_ARGS})
#message(STATUS GIT_EXECUTABLE=${GIT_EXECUTABLE})

# include(ExternalProject)

# SET(BTK_CMAKE_ARGS
#   -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
#   -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
#   -DANN_INCLUDE_DIR:PATH=${ANN_INCLUDE_DIR}
#   -DANN_LIBRARY:FILEPATH=${ANN_LIBRARY}
#   -DTCLAP_DIRECTORY:PATH=${TCLAP_DIRECTORY}
#   -DTCL_LIBRARY:FILEPATH=${TCL_LIBRARY}
#   -DTK_LIBRARY:FILEPATH=${TK_LIBRARY}
#   -DITK_DIR:PATH=${ITK_DIR}
#   -DVTK_DIR:PATH=${VTK_DIR}
#   -DUSE_OMP:BOOL=${USE_OMP}
#   -DBUILD_EXPERIMENTAL:BOOL=OFF
#   -DBUILD_TESTING:BOOL=OFF
#   -DBUILD_DOCUMENTATION:BOOL=OFF
#   -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}

# )

# ExternalProject_Add(fbrain
#   PREFIX ${BTK_BIN_DIR}
#   SOURCE_DIR ${BTK_SRC_DIR}
#   BINARY_DIR ${BTK_BIN_DIR}
#   DOWNLOAD_DIR ${BTK_BIN_DIR}/download
#   TMP_DIR ${BTK_BIN_DIR}/tmp
#   STAMP_DIR ${BTK_BIN_DIR}/stamp
#   DOWNLOAD_COMMAND ""
#   INSTALL_COMMAND ""
#   CMAKE_ARGS ${BTK_CMAKE_ARGS}
# )

add_subdirectory(Denoising)
add_subdirectory(Evaluation)
# add_subdirectory(IntensityNormalization)
# add_subdirectory(BrainLocalizationExtraction)
# add_subdirectory(Reconstruction)
add_subdirectory(Tools)

#add_subdirectory(Test)
# add_subdirectory(CRKit/Segmentation)
# add_subdirectory(CRKit/FetalInitializer)
# add_subdirectory(CRKit/FetalL1normInitializer)
# add_subdirectory(CRKit/FetalBrainLocalization)
# add_subdirectory(CRKit/FetalInitializer2)
# add_subdirectory(CRKit/FetalInitializer3)
# add_subdirectory(CRKit/Tools)
# add_subdirectory(CRKit/STAPLE)
# add_subdirectory(CRKit/Registration)
#add_subdirectory(Code)


