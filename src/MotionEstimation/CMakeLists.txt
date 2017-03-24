PROJECT(MotionEstimation)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(
    ${TCLAP_DIRECTORY}
)

#---- Set BTK library DIR ---------------------------------------------------------------------
#LINK_DIRECTORIES(${BTK_BIN_DIR}/Code)

# ---- Reconstruction -------------------------------------------------------------------
# BTK
INCLUDE_DIRECTORIES(
    ${ANN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Code/BTK
)
#MIALSRTK
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/Code/Reconstruction
	${CMAKE_SOURCE_DIR}/Code/Registration
    ${CMAKE_SOURCE_DIR}/Code/Transformations
)

ADD_EXECUTABLE(mialsrtkImageReconstruction mialsrtkImageReconstruction.cxx
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/mialsrtkLowToHighImageResolutionMethod.h
    ${CMAKE_SOURCE_DIR}/Code/Registration/mialsrtkSliceBySliceRigidRegistration.h
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/mialsrtkResampleImageByInjectionFilter.h
    ${CMAKE_SOURCE_DIR}/Code/Transformations/mialsrtkVersorSliceBySliceTransform.h
    ${CMAKE_SOURCE_DIR}/Code/Transformations/mialsrtkSliceBySliceTransform.h
    ${CMAKE_SOURCE_DIR}/Code/Reconstruction/mialsrtkImageIntersectionCalculator.h
    ${CMAKE_SOURCE_DIR}/Code/Registration/mialsrtkRegistration.h
    ${CMAKE_SOURCE_DIR}/Code/Registration/mialsrtkRigidRegistration.h
)
TARGET_LINK_LIBRARIES(mialsrtkImageReconstruction ${ITK_LIBRARIES})
#TARGET_LINK_LIBRARIES(mialsrtkImageReconstruction btkToolsLibrary ${ITK_LIBRARIES})

INSTALL(TARGETS
                mialsrtkImageReconstruction
        DESTINATION bin)
