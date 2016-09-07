cmake_minimum_required(VERSION 2.8)

# Set module path for findOpenCL
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake)

# Including the APIs
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../discovery_protocol/api/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include/OpenCL/)

# Including Opencl
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

# Paths for kernels and cl_extension (used in the kernels)
add_definitions(-DCL_ACTIVE_GROUP_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../../discovery_protocol/api/)
add_definitions(-DKERNEL_DIR=${PROJECT_BINARY_DIR}/bin/kernels/)

# Define int and atomic int on the host side
add_definitions(-DINT_TYPE=cl_int)
add_definitions(-DATOMIC_INT_TYPE=cl_int)

# Include pannotia commons
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/graph_parser/)

# Add pannotia librarys
add_library(graph_parse 
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_parser/parse.cpp)

add_library(util 
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_parser/util.cpp)

# Make a bin directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# --sssp apps--

# -sssp
add_executable(sssp 
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/sssp/sssp.cpp
)
target_link_libraries(sssp graph_parse util ${OPENCL_LIBRARIES})

# -sssp-gb
add_executable(sssp-gb
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/sssp/sssp_gb.cpp
)
target_link_libraries(sssp-gb graph_parse util ${OPENCL_LIBRARIES})

# --bc app--

# -bc
add_executable(bc
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/bc/bc.cpp
)
target_link_libraries(bc graph_parse util ${OPENCL_LIBRARIES})

# -bc-gb
add_executable(bc-gb
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/bc/bc_gb.cpp
)
target_link_libraries(bc-gb graph_parse util ${OPENCL_LIBRARIES})

# --color app--

# -color
add_executable(color
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/color/color.cpp
)
target_link_libraries(color graph_parse util ${OPENCL_LIBRARIES})

# -color-gb
add_executable(color-gb
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/color/color_gb.cpp
)
target_link_libraries(color-gb graph_parse util ${OPENCL_LIBRARIES})

# --mis app--

# -mis
add_executable(mis
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/mis/mis.cpp
)
target_link_libraries(mis graph_parse util ${OPENCL_LIBRARIES})

# -mis-gb
add_executable(mis-gb
  ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/mis/mis_gb.cpp
)
target_link_libraries(mis-gb graph_parse util ${OPENCL_LIBRARIES})

# Copy kernels to their own directorys
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/sssp/kernel/sssp_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/bc/kernel/bc_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/color/kernel/color_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/graph_app/mis/kernel/mis_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)