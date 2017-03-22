PROJECT(Utilities)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(${TCLAP_DIRECTORY})
INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/BTK)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Reconstruction)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Transformations)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Registration)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)

ADD_EXECUTABLE(mialsrtkComputeVolume mialsrtkComputeVolume.cxx)
TARGET_LINK_LIBRARIES(mialsrtkComputeVolume ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkCropImageUsingMask mialsrtkCropImageUsingMask.cxx)
TARGET_LINK_LIBRARIES(mialsrtkCropImageUsingMask ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkCreateImageWithHalo mialsrtkCreateImageWithHalo.cxx)
TARGET_LINK_LIBRARIES(mialsrtkCreateImageWithHalo  ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkCreateWhiteMatterVolumeFromLabels mialsrtkCreateWhiteMatterVolumeFromLabels.cxx)
TARGET_LINK_LIBRARIES(mialsrtkCreateWhiteMatterVolumeFromLabels  ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkFourierTransform mialsrtkFourierTransform.cxx)
TARGET_LINK_LIBRARIES(mialsrtkFourierTransform ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkMaskImage mialsrtkMaskImage.cxx)
TARGET_LINK_LIBRARIES(mialsrtkMaskImage ${ITK_LIBRARIES})


INSTALL(TARGETS 
				mialsrtkCreateImageWithHalo
				mialsrtkCreateWhiteMatterVolumeFromLabels
				mialsrtkCropImageUsingMask				
				mialsrtkCropImageUsingMask
				mialsrtkFourierTransform
				mialsrtkMaskImage
        DESTINATION bin)

