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
