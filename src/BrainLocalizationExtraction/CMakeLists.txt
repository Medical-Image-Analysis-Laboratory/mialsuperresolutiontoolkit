PROJECT(BrainLocalizationExtraction)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(${TCLAP_DIRECTORY})
INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/CRKit)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Transformations)

ADD_EXECUTABLE(mialsrtkSegmentationWeightedFusion mialsrtkSegmentationWeightedFusion.cxx)
TARGET_LINK_LIBRARIES(mialsrtkSegmentationWeightedFusion ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkSegmentationWeightedFusionWithUnanimousConsensus mialsrtkSegmentationWeightedFusionWithUnanimousConsensus.cxx)
TARGET_LINK_LIBRARIES(mialsrtkSegmentationWeightedFusionWithUnanimousConsensus ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkRefineHRMaskByIntersection mialsrtkRefineHRMaskByIntersection.cxx)
TARGET_LINK_LIBRARIES(mialsrtkRefineHRMaskByIntersection ${ITK_LIBRARIES})



INSTALL(TARGETS
		mialsrtkRefineHRMaskByIntersection 
		mialsrtkSegmentationWeightedFusion
		mialsrtkSegmentationWeightedFusionWithUnanimousConsensus				
        DESTINATION bin)

