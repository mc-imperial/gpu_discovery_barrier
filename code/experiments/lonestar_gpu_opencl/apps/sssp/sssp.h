/** Single source shortest paths -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Single source shortest paths.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// Common functions for both sssp_port and sssp_non_port. OpenCL by
// Tyler Sorensen (2016)

// Initialise the OpenCL utilities
void init_opencl() {
  int err;
  device = create_device();
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERR(err);
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERR(err);
  char opts[500];
  get_compile_opts_wgs(opts, wgs);
  prog = build_program(context, device, CL_FILE, opts);
}

// Write solution to file
void print_output(const char *filename, cl_command_queue *q, cl_foru *hdist, cl_mem *dist, Graph graph) {
  int err;

  err = clEnqueueReadBuffer(*q,
                            *dist,
                            1,
                            0,
                            sizeof(cl_foru)*graph.nnodes,
                            hdist,
                            0,
                            0,
                            0);

  printf("Writing output to %s\n", filename);
  FILE *o = fopen(filename, "w");

  for (int i = 0; i < graph.nnodes; i++) {
    fprintf(o, "%d: %d\n", i, hdist[i]);
  }
  fclose(o);
}

// Clean up the OpenCL utilities
void clean_opencl() {
  clReleaseCommandQueue(queue);
  clReleaseProgram(prog);
  clReleaseContext(context);
}

// Simple function to check if a value is a power of 2
int pow_2(int x) {
  return ((x != 0) && !(x & (x - 1)));
}
