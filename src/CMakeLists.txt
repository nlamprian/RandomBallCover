include_directories ( ${CLUtils_INCLUDE_DIR} )

add_library ( RBCHelperFuncs STATIC RBC/tests/helper_funcs.cpp )
add_library ( RBCAlgorithms STATIC RBC/algorithms.cpp )

add_dependencies ( RBCHelperFuncs CLUtils )
add_dependencies ( RBCAlgorithms  CLUtils )

target_include_directories ( 
    RBCHelperFuncs PUBLIC 
    ${COMMON_INCLUDES} 
)

target_include_directories ( 
    RBCAlgorithms PUBLIC 
    ${COMMON_INCLUDES} 
)

install ( DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include )
install ( DIRECTORY ${PROJECT_BINARY_DIR}/lib/ DESTINATION lib/RBC )

# ==============================================================================

find_package ( Freenect QUIET )

if ( FREENECT_FOUND )

    find_package ( GLUT REQUIRED )
    find_package ( GLEW REQUIRED )
    find_package ( libusb-1.0 REQUIRED )
    find_package ( Threads REQUIRED )

    include_directories ( ${OPENGL_INCLUDE_DIRS} 
                          ${GLUT_INCLUDE_DIRS} 
                          ${GLEW_INCLUDE_DIRS} 
                          ${LIBUSB_1_INCLUDE_DIRS} 
                          ${FREENECT_INCLUDE_DIR} )

    add_executable ( grub_kinect_frames grub_kinect_frames.cpp )

    target_link_libraries ( grub_kinect_frames ${OPENGL_LIBRARIES} 
                                               ${GLUT_LIBRARY} 
                                               ${GLEW_LIBRARIES} 
                                               ${FREENECT_LIBRARIES} 
                                               ${OPENCL_LIBRARIES} 
                                               ${CMAKE_THREAD_LIBS_INIT} )

endif ( FREENECT_FOUND )
