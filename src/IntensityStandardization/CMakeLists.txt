PROJECT(Utilities)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(${TCLAP_DIRECTORY})
INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/BTK)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Reconstruction)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Transformations)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Registration)

ADD_EXECUTABLE(mialsrtkCorrectBiasFieldWithMotionApplied mialsrtkCorrectBiasFieldWithMotionApplied.cxx)
TARGET_LINK_LIBRARIES(mialsrtkCorrectBiasFieldWithMotionApplied ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkCorrectBiasField mialsrtkCorrectBiasField.cxx)
TARGET_LINK_LIBRARIES(mialsrtkCorrectBiasField ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkCorrectSliceIntensity mialsrtkCorrectSliceIntensity.cxx)
TARGET_LINK_LIBRARIES(mialsrtkCorrectSliceIntensity ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkSliceBySliceCorrectBiasField mialsrtkSliceBySliceCorrectBiasField.cxx)
TARGET_LINK_LIBRARIES(mialsrtkSliceBySliceCorrectBiasField ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkN4BiasFieldCorrection mialsrtkN4BiasFieldCorrection.cxx)
TARGET_LINK_LIBRARIES(mialsrtkN4BiasFieldCorrection ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkSliceBySliceN4BiasFieldCorrection mialsrtkSliceBySliceN4BiasFieldCorrection.cxx)
TARGET_LINK_LIBRARIES(mialsrtkSliceBySliceN4BiasFieldCorrection ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkIntensityStandardization mialsrtkIntensityStandardization.cxx)
TARGET_LINK_LIBRARIES(mialsrtkIntensityStandardization  ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkImageProjectionToPositiveIntensities mialsrtkImageProjectionToPositiveIntensities.cxx)
TARGET_LINK_LIBRARIES(mialsrtkImageProjectionToPositiveIntensities  ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkIntensityRescaleToMaxInAllImages mialsrtkIntensityRescaleToMaxInAllImages.cxx)
TARGET_LINK_LIBRARIES(mialsrtkIntensityRescaleToMaxInAllImages  ${ITK_LIBRARIES})

INSTALL(TARGETS 
				mialsrtkCorrectBiasField
				mialsrtkCorrectBiasFieldWithMotionApplied
				mialsrtkCorrectSliceIntensity
				mialsrtkIntensityStandardization
				mialsrtkN4BiasFieldCorrection
				mialsrtkSliceBySliceN4BiasFieldCorrection
				mialsrtkSliceBySliceCorrectBiasField
				mialsrtkImageProjectionToPositiveIntensities
				mialsrtkIntensityRescaleToMaxInAllImages
        DESTINATION bin)

