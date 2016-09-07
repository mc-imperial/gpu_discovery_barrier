/** Breadth-first search -*- C++ -*-
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
 * Example breadth-first search application for demoing Galois system.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// Common functions between bfs-port and bfs-non-port

// Write solution to file
void write_solution(const char *fname, Graph *hgraph, cl_mem *dist) {
  int err;
  foru *h_dist;
  h_dist = (foru *) malloc(hgraph->nnodes * sizeof(foru));
  printf("number of nodes: %d\n", hgraph->nnodes);
  assert(h_dist != NULL);

  err = clEnqueueReadBuffer(queue,
                            *dist,
                            1,
                            0,
                            sizeof(cl_int) * hgraph->nnodes,
                            h_dist,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  printf("Writing solution to %s\n", fname);
  FILE *f = fopen(fname, "w");

  // Formatted like Merrill's code for comparison
  fprintf(f, "Computed solution (source dist): [");

  for (int node = 0; node < hgraph->nnodes; node++) {
    fprintf(f, "%d:%d\n ", node, h_dist[node]);
  }

  fprintf(f, "]");
  fclose(f);
  free(h_dist);
}

// Init OpenCL utilities
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

// Clean OpenCL utilities
void clean_opencl() {
  clReleaseCommandQueue(queue);
  clReleaseProgram(prog);
  clReleaseContext(context);
}

// Simple function to check if a value is a power of 2
int pow_2(int x) {
  return ((x != 0) && !(x & (x - 1)));
}
