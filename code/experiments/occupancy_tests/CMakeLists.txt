cmake_minimum_required(VERSION 2.8)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../discovery_protocol/api/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include/OpenCL/)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

find_package(OpenCL REQUIRED)

include_directories(${OPENCL_INCLUDE_DIR})

set (EXECUTABLE_NAME "occupancy_test")

add_executable(${EXECUTABLE_NAME} 
  src/occupancy_test.c
)

set (EXECUTABLE_NAME2 "device_query")

add_executable(${EXECUTABLE_NAME2} 
  src/device_query.c
)

set (EXECUTABLE_NAME3 "time_prot")

add_executable(${EXECUTABLE_NAME3} 
  src/time_prot.c
)


add_definitions(-DCL_ACTIVE_GROUP_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../../discovery_protocol/api/)
add_definitions(-DKERNEL_DIR=${PROJECT_BINARY_DIR}/bin/kernels/)
add_definitions(-DINT_TYPE=cl_int)
add_definitions(-DATOMIC_INT_TYPE=cl_int)


target_link_libraries(${EXECUTABLE_NAME} ${OPENCL_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME2} ${OPENCL_LIBRARIES})
target_link_libraries(${EXECUTABLE_NAME3} ${OPENCL_LIBRARIES})

file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/kernels)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/occupancy_test.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)