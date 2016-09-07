// A simple intra-block OpenCL scan function
// From: http://www.nehalemlabs.net/prototype/blog/2014/06/23/parallel-programming-with-opencl-and-python-parallel-scan/
// Requires the number of threads per workgroups to be a power-of-2

#pragma once

void block_int_exclusive_sum_scan(__local int* b, __local int *tmp, int input, int *output, int *total_edges, int n_items) {

  int lid = get_local_id(0);
  b[lid] = input;
  barrier(CLK_LOCAL_MEM_FENCE);
  int dp = 1;

  for (uint s = n_items >> 1; s > 0; s >>= 1) {

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < s) {
      uint i = dp*(2*lid+1)-1;
      uint j = dp*(2*lid+2)-1;
      b[j] += b[i];
    }
    dp <<= 1;
  }

  if (lid == 0)
    b[n_items - 1] = 0;

  for (uint s = 1; s < n_items; s <<= 1) {

    dp >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < s) {
      uint i = dp*(2*lid+1)-1;
      uint j = dp*(2*lid+2)-1;

      int t = b[j];
      b[j] += b[i];
      b[i] = t;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  *output = b[lid];

  if (lid == n_items - 1) {
    *tmp = b[n_items - 1];
    *tmp += input;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  *total_edges = *tmp;
  return;
}
