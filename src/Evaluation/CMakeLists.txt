PROJECT(Evaluation)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(${TCLAP_DIRECTORY})
INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)

ADD_EXECUTABLE(mialsrtkComputeHistogramSpread mialsrtkComputeHistogramSpread.cxx)
IF(CMAKE_SYSTEM_NAME STREQUAL Linux)
    TARGET_LINK_LIBRARIES(mialsrtkComputeHistogramSpread ${ITK_LIBRARIES} rt)
ENDIF(CMAKE_SYSTEM_NAME STREQUAL Linux)
IF(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    TARGET_LINK_LIBRARIES(mialsrtkComputeHistogramSpread ${ITK_LIBRARIES})
ENDIF(CMAKE_SYSTEM_NAME STREQUAL Darwin)


ADD_EXECUTABLE(mialsrtkEvaluateContrast mialsrtkEvaluateContrast.cxx)
TARGET_LINK_LIBRARIES(mialsrtkEvaluateContrast ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkComputeImageMedianCNR mialsrtkComputeImageMedianCNR.cxx)
TARGET_LINK_LIBRARIES(mialsrtkComputeImageMedianCNR ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkComputeHausdorffDistance mialsrtkComputeHausdorffDistance.cxx)
TARGET_LINK_LIBRARIES(mialsrtkComputeHausdorffDistance ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkEvaluateLabelOverlapMeasures mialsrtkEvaluateLabelOverlapMeasures.cxx)
TARGET_LINK_LIBRARIES(mialsrtkEvaluateLabelOverlapMeasures ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkEvaluateSharpnessMeasures mialsrtkEvaluateSharpnessMeasures.cxx)
TARGET_LINK_LIBRARIES(mialsrtkEvaluateSharpnessMeasures ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkEvaluateReconstructionQualityMeasures mialsrtkEvaluateReconstructionQualityMeasures.cxx)
TARGET_LINK_LIBRARIES(mialsrtkEvaluateReconstructionQualityMeasures ${ITK_LIBRARIES} ${VTK_LIBRARIES})

INSTALL(TARGETS 
				mialsrtkComputeHausdorffDistance
				mialsrtkComputeHistogramSpread
				mialsrtkComputeImageMedianCNR
				mialsrtkEvaluateContrast
				mialsrtkEvaluateLabelOverlapMeasures
                mialsrtkEvaluateSharpnessMeasures
                mialsrtkEvaluateReconstructionQualityMeasures
				
        DESTINATION bin)

