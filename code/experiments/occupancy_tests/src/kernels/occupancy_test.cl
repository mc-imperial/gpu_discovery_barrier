// kernels for testing occupancy.

#include "discovery.cl"

// This kernel executes the protocol followed by a barrier. 
// The barrier exists for finding the true occupancy bound
// when the protocol is skipped. 
__kernel void run_test(__global discovery_kernel_ctx *gl_ctx, __local void* loc_mem) {
    DISCOVERY_PROTOCOL(gl_ctx);  
    discovery_barrier(gl_ctx, &local_ctx);
}

// This kernel simply runs the protocol. This is for timing
// the protocol.
__kernel void run_prot(__global discovery_kernel_ctx *gl_ctx, __local void* loc_mem) {
  DISCOVERY_PROTOCOL(gl_ctx);
}
//
