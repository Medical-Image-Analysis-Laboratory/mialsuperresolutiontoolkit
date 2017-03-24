PROJECT(Applications)
INCLUDE_REGULAR_EXPRESSION("^.*$")

#message(STATUS BTK_SRC_DIR=${BTK_SRC_DIR})

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

#MIALTK
INCLUDE_DIRECTORIES(
	${CMAKE_SOURCE_DIR}/Code/Maths
	${CMAKE_SOURCE_DIR}/Code/Evaluation
	${CMAKE_SOURCE_DIR}/Code/Reconstruction
	${CMAKE_SOURCE_DIR}/Code/Registration
        ${CMAKE_SOURCE_DIR}/Code/Transformations
	${CMAKE_SOURCE_DIR}/Code/Timer
)


ADD_EXECUTABLE(mialtkTVSuperResolution mialtkTVSuperResolution.cxx)
TARGET_LINK_LIBRARIES(mialtkTVSuperResolution ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialtkRobustTVSuperResolutionWithGMM mialtkRobustTVSuperResolutionWithGMM.cxx)
TARGET_LINK_LIBRARIES(mialtkRobustTVSuperResolutionWithGMM ${ITK_LIBRARIES})


INSTALL(TARGETS
		mialtkTVSuperResolution
                mialtkRobustTVSuperResolutionWithGMM
        DESTINATION bin)
