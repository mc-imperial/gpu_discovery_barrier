// The host part of an OpenCL port the Xaio/Feng GPU global barrier
// using OpenCL 2.0 atomics. This is an unsafe implementation. That
// is, it attempts to synchronise across all workgroups on the GPU.

// A safe variant (using the discovery protocol) is available in the
// discovery_protocol directory

// Port by Tyler Sorensen (2016)

// Call this function to create and initialise a global barrier cl_mem
// object
int allocate_and_init_gbar(cl_context *c, cl_command_queue *q, cl_mem *gbar_arr, int wgn) {

  int err;

  // Create the flag array
  *gbar_arr = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_int)*wgn, NULL ,&err);

  if (err != 0) { return err; }

  // Initialise a host side flag array to copy to the device
  cl_int *zero_array = (int *) malloc(sizeof(cl_int) *wgn);
  for (int i = 0; i < wgn; i++) {
    zero_array[i] = 0;
  }

  // Copy values to the device
  err = clEnqueueWriteBuffer(*q,
                             *gbar_arr,
                             1,
                             0,
                             sizeof(cl_int)*wgn,
                             zero_array,
                             0,
                             0,
                             0);

  if (err != 0) { return err; }

  // Free the temporary host flag array
  free(zero_array);
  return 0;
}
