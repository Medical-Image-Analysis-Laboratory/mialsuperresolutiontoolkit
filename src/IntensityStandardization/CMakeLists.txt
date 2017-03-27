PROJECT(IntensityStandardization)
INCLUDE_REGULAR_EXPRESSION("^.*$")

INCLUDE_DIRECTORIES(${TCLAP_DIRECTORY})
INCLUDE_DIRECTORIES(${ANN_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/BTK)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Reconstruction)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Transformations)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Maths)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Registration)

#Find Python
find_program(PYTHON "python")

if (PYTHON)
#    set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
#    set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/pymialsrtkHistogramNormalization/setup.py")
#    set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/pymialsrtkHistogramNormalization/__init__.py")
#    set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")

#    configure_file(${SETUP_PY_IN} ${SETUP_PY})

#    add_custom_command(OUTPUT ${OUTPUT}
#                       COMMAND ${PYTHON} ${SETUP_PY} build
#                       COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
#                       DEPENDS ${DEPS})

#    add_custom_target(mialsrtkHistogramNormalization ALL DEPENDS ${OUTPUT})

#    install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")
    set(CONFIG_PY_DIR ${CMAKE_CURRENT_BINARY_DIR})
    configure_file(mialsrtkHistogramNormalization.py.in ${CMAKE_CURRENT_BINARY_DIR}/mialsrtkHistogramNormalization.py @ONLY)
    set(CONFIG_PY_DIR ${PYTHON_INSTALL_DIR})
    configure_file(mialsrtkHistogramNormalization.py.in
    ${PROJECT_BINARY_DIR}/installFiles/program.py @ONLY)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/mialsrtkHistogramNormalization.py
            DESTINATION
                    bin)
endif()

#set(CONFIG_PY_DIR ${PROJECT_SOURCE_DIR})
#configure_file(program.py.in ${PROJECT_BINARY_DIR}/program.py @ONLY)
#set(CONFIG_PY_DIR ${PYTHON_INSTALL_DIR})
#configure_file(program.py.in
#  ${PROJECT_BINARY_DIR}/installFiles/program.py @ONLY)
#install(PROGRAMS ${PROJECT_BINARY_DIR}/installFiles/program.py
#  DESTINATION bin)

#FIND_PACKAGE(PythonLibs)
#INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

message(STATUS PYTHON_INCLUDE_PATH=${PYTHON_INCLUDE_PATH})

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

