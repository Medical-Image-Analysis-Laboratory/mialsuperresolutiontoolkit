PROJECT(MotionEstimation)
INCLUDE_REGULAR_EXPRESSION("^.*$")

#message(STATUS BTK_SRC_DIR=${BTK_SRC_DIR})

INCLUDE_DIRECTORIES(
    ${TCLAP_DIRECTORY}
    ${BTK_SRC_DIR}/Code/Tools
    ${BTK_SRC_DIR}/Code/Maths
    ${BTK_SRC_DIR}/Code/Transformations
    ${BTK_SRC_DIR}/Code/Visualization

)

#---- Set BTK library DIR ---------------------------------------------------------------------
LINK_DIRECTORIES(${BTK_BIN_DIR}/Code)

# ---- Denoising ------------------------------------------------------------------------

INCLUDE_DIRECTORIES(${BTK_SRC_DIR}/Code/Denoising)


# ---- Reconstruction -------------------------------------------------------------------
# BTK
INCLUDE_DIRECTORIES(
    ${ANN_INCLUDE_DIR}
    ${BTK_SRC_DIR}/Code/Reconstruction
    ${BTK_SRC_DIR}/Code/Reconstruction/SuperResolution
    ${BTK_SRC_DIR}/Code/Diffusion
    ${BTK_SRC_DIR}/Code/Segmentation
    ${BTK_SRC_DIR}/Code/Transformations
    ${BTK_SRC_DIR}/Code/Registration
    ${BTK_SRC_DIR}/Code/Maths
    ${BTK_SRC_DIR}/Code/Visualization
    ${BTK_SRC_DIR}/Code/ImageFilters
)
#MIALTK
INCLUDE_DIRECTORIES(
  ${CMAKE_SOURCE_DIR}/Code/Maths
	${CMAKE_SOURCE_DIR}/Code/Evaluation
	${CMAKE_SOURCE_DIR}/Code/Reconstruction
	${CMAKE_SOURCE_DIR}/Code/Registration
        ${CMAKE_SOURCE_DIR}/Code/Transformations
	${CMAKE_SOURCE_DIR}/Code/Registration/CostFunctions
	${CMAKE_SOURCE_DIR}/Code/Registration/Optimizers
)



ADD_EXECUTABLE(mialtkImageReconstruction mialtkImageReconstruction.cxx
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/mialtkLowToHighImageResolutionMethod.h
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/btkLowToHighImageResolutionMethod.h
    ${CMAKE_SOURCE_DIR}/Code/Registration/mialtkSliceBySliceRigidRegistration.h
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/mialtkResampleImageByInjectionFilter.h
    ${CMAKE_SOURCE_DIR}/Code/Transformations/mialtkVersorSliceBySliceTransform.h
    ${CMAKE_SOURCE_DIR}/Code/Transformations/mialtkSliceBySliceTransform.h
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/mialtkImageIntersectionCalculator.h
    ${CMAKE_SOURCE_DIR}/Code/Registration/mialtkRegistration.h
    ${CMAKE_SOURCE_DIR}/Code/Registration/mialtkRigidRegistration.h
)
TARGET_LINK_LIBRARIES(mialtkImageReconstruction btkToolsLibrary ${ITK_LIBRARIES})




INSTALL(TARGETS
                mialtkImageReconstruction
        DESTINATION bin)
