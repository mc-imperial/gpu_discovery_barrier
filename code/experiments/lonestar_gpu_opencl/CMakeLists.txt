cmake_minimum_required(VERSION 2.8)

# Set module path for findOpenCL
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../common/cmake)

# Including the discovery protocol and opencl utilities
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../discovery_protocol/api/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common/include/OpenCL/)

# Including actual Opencl
find_package(OpenCL REQUIRED)
include_directories(${OPENCL_INCLUDE_DIR})

# Paths for kernels and device discovery protocol
add_definitions(-DCL_ACTIVE_GROUP_PATH=${CMAKE_CURRENT_SOURCE_DIR}/../../discovery_protocol/api/)
add_definitions(-DKERNEL_DIR=${PROJECT_BINARY_DIR}/bin/kernels/)
add_definitions(-DLONESTAR_CL_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/include/)

# Define int and atomic int on the host side
add_definitions(-DINT_TYPE=cl_int)
add_definitions(-DATOMIC_INT_TYPE=cl_int)

# Include GPU-Lonestar common
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

# Make a bin directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# --bfs applications

# bfs portable
add_executable(bfs-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/bfs/bfs_port.cpp
)
target_link_libraries(bfs-port ${OPENCL_LIBRARIES})

# bfs non-portable
add_executable(bfs-non-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/bfs/bfs_non_port.cpp
)
target_link_libraries(bfs-non-port ${OPENCL_LIBRARIES})

# --mst applications

# mst portable
add_executable(mst-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/mst/mst_port.cpp
)
target_link_libraries(mst-port ${OPENCL_LIBRARIES})

# mst non-portable
add_executable(mst-non-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/mst/mst_non_port.cpp
)
target_link_libraries(mst-non-port ${OPENCL_LIBRARIES})

# --sssp applications

# sssp portable
add_executable(sssp-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/sssp/sssp_port.cpp
)
target_link_libraries(sssp-port ${OPENCL_LIBRARIES})

# sssp non-portable
add_executable(sssp-non-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/sssp/sssp_non_port.cpp
)
target_link_libraries(sssp-non-port ${OPENCL_LIBRARIES})

# --dmr applications

# dmr portable
add_executable(dmr-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/dmr/dmr_port.cpp
)
target_link_libraries(dmr-port ${OPENCL_LIBRARIES})

# dmr non-portable
add_executable(dmr-non-port
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/dmr/dmr_non_port.cpp
)
target_link_libraries(dmr-non-port ${OPENCL_LIBRARIES})

#copy kernels to their own directorys
file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/bin/kernels)

# bfs kernel files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/bfs/kernel/bfs_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/bfs/kernel/bfs_non_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/bfs/kernel/bfs_common.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)

# mst kernel files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/mst/kernel/mst_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/mst/kernel/mst_non_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/mst/kernel/mst_common.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)

# sssp kernel files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/sssp/kernel/sssp_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/sssp/kernel/sssp_non_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/sssp/kernel/sssp_common.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)

# dmr kernel files
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/dmr/kernel/dmr_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/dmr/kernel/dmr_non_port_kernel.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/apps/dmr/kernel/dmr_common.cl DESTINATION ${PROJECT_BINARY_DIR}/bin/kernels)


