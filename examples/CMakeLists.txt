find_package ( OpenCV QUIET )
find_package ( PLplot QUIET )

include_directories ( ${CLUtils_INCLUDE_DIR} )

if ( OpenCV_FOUND )

    include_directories ( ${OpenCV_INCLUDE_DIR} )

    add_executable ( ${FNAME}_img_segmentation img_segmentation.cpp )

    add_dependencies ( ${FNAME}_img_segmentation CLUtils )

    target_link_libraries ( ${FNAME}_img_segmentation ${CLUtils_LIBRARIES} 
                                                      RBCAlgorithms 
                                                      ${OPENGL_LIBRARIES} 
                                                      ${OPENCL_LIBRARIES}
                                                      ${OpenCV_LIBRARIES} )

endif ( OpenCV_FOUND )

if ( PLplot_FOUND )

    include_directories ( ${PLplot_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_2d_plot 2d_plot.cpp )

    add_dependencies ( ${FNAME}_2d_plot CLUtils )

    target_link_libraries ( ${FNAME}_2d_plot ${CLUtils_LIBRARIES} 
                                             RBCAlgorithms 
                                             ${OPENGL_LIBRARIES} 
                                             ${OPENCL_LIBRARIES}
                                             ${PLplot_LIBRARIES} )

endif ( PLplot_FOUND )
