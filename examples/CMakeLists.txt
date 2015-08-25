find_package ( OpenCV QUIET )
find_package ( PLplot QUIET )

include_directories ( ${CLUtils_INCLUDE_DIR} )

if ( OpenCV2_FOUND )

    include_directories ( ${OpenCV2_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_img_segmentation img_segmentation.cpp )
    add_executable ( ${FNAME}_rgb_img_segmentation rgb_img_segmentation.cpp )

    add_dependencies ( ${FNAME}_img_segmentation CLUtils )
    add_dependencies ( ${FNAME}_rgb_img_segmentation CLUtils )

    target_link_libraries ( ${FNAME}_img_segmentation ${CLUtils_LIBRARIES} 
                                                      RBCAlgorithms 
                                                      ${OPENGL_LIBRARIES} 
                                                      ${OPENCL_LIBRARIES}
                                                      ${OpenCV2_LIBRARIES} )

    target_link_libraries ( ${FNAME}_rgb_img_segmentation ${CLUtils_LIBRARIES} 
                                                          RBCAlgorithms 
                                                          ${OPENGL_LIBRARIES} 
                                                          ${OPENCL_LIBRARIES}
                                                          ${OpenCV2_LIBRARIES} )

endif ( OpenCV2_FOUND )

if ( PLplot_FOUND )

    include_directories ( ${PLplot_INCLUDE_DIRS} )

    add_executable ( ${FNAME}_2d_plot 2d_plot.cpp )
    add_executable ( ${FNAME}_2d_plot_nn 2d_plot_nn.cpp )

    add_dependencies ( ${FNAME}_2d_plot CLUtils )
    add_dependencies ( ${FNAME}_2d_plot_nn CLUtils )

    target_link_libraries ( ${FNAME}_2d_plot ${CLUtils_LIBRARIES} 
                                             RBCAlgorithms 
                                             ${OPENGL_LIBRARIES} 
                                             ${OPENCL_LIBRARIES}
                                             ${PLplot_LIBRARIES} )

    target_link_libraries ( ${FNAME}_2d_plot_nn ${CLUtils_LIBRARIES} 
                                                RBCAlgorithms 
                                                ${OPENGL_LIBRARIES} 
                                                ${OPENCL_LIBRARIES}
                                                ${PLplot_LIBRARIES} )

endif ( PLplot_FOUND )
