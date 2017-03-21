PROJECT(Denoising)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(${TCLAP_DIRECTORY})
INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Reconstruction)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Transformations)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Registration)
# INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)


ADD_EXECUTABLE(mialsrtkTVDenoising mialtkTVDenoising.cxx)
TARGET_LINK_LIBRARIES(mialsrtkTVDenoising  ${ITK_LIBRARIES})

ADD_EXECUTABLE(mialsrtkGaussianDenoising mialtkGaussianDenoising.cxx)
TARGET_LINK_LIBRARIES(mialsrtkGaussianDenoising  ${ITK_LIBRARIES})


INSTALL(TARGETS 
		mialsrtkGaussianDenoising
                mialsrtkTVDenoising
        DESTINATION 
		bin)