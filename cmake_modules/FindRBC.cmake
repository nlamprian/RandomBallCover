# Try to find RBC
# Once done this will define:
#
#   RBC_ROOT         - if set, it will try to find in this folder
#   RBC_FOUND        - system has RBC
#   RBC_INCLUDE_DIR  - the RBC include directory
#   RBC_LIBRARIES    - link these to use RBC

find_path ( 
    RBC_INCLUDE_DIR
    NAMES common.hpp data_types.hpp algorithms.hpp tests/helper_funcs.hpp 
    HINTS ${RBC_ROOT}/include 
          /usr/include
          /usr/local/include 
    DOC "The directory where RBC headers reside"
)

find_library ( 
    RBC_LIBRARY 
    NAMES RBCAlgorithms RBCHelperFuncs  
    PATHS ${RBC_ROOT}/build/lib 
          /usr/lib 
          /usr/local/lib 
    DOC "The directory where RBC libraries reside"
)

include ( FindPackageHandleStandardArgs )

find_package_handle_standard_args ( 
    RBC 
    FOUND_VAR RBC_FOUND
    REQUIRED_VARS RBC_INCLUDE_DIR RBC_LIBRARY 
)

if ( RBC_FOUND )
    set ( RBC_LIBRARIES ${RBC_LIBRARY} )
else ( RBC_FOUND )
    set ( RBC_LIBRARIES )
endif ( RBC_FOUND )

mark_as_advanced ( 
    RBC_INCLUDE_DIR
    RBC_LIBRARY
)
