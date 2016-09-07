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

// The GPU-Lonestar sssp-wlc application ported to OpenCL.  Port by
// Tyler Sorensen (2016)

// This application uses a non-portable inter-workgroup barrier which
// requires knowledge about the target GPU occupancy

// A high-level OpenCL check_err function
#define CHECK_ERR(err)  if (err < 0 ) { printf("CL_ERROR %s %d (%d)\n", __FILE__, __LINE__, err); exit(1); }

typedef unsigned foru;

#include "my_opencl.h"
#include "common.h"
#include "header.h"
#include "graph.h"
#include "kernelconf.h"
#include "gbar.h"
#include "worklistc.h"

typedef cl_uint cl_foru;

// OpenCL utilities
cl_context context;
cl_device_id device;
cl_command_queue queue;
cl_program prog;
int wgs;
int wgn;

// Number of vertices
unsigned int NVERTICES;

const char * CL_FILE = STRINGIFY(KERNEL_DIR) "sssp_non_port_kernel.cl";

#include "sssp.h"

// Top level sssp function
void sssp(foru *hdist, cl_mem *dist, Graph *hgraph, cl_mem *dgraph, KernelConfig *kconf) {

  foru foruzero = 0.0;
  int err;
  unsigned int NBLOCKS, FACTOR = 128;
  int iteration = 0;

  cl_kernel drelax2;

  double starttime, endtime;
  double runtime;

  // Initialise distance to zero
  err = clEnqueueWriteBuffer(queue,
                             *dist,
                             1,
                             0,
                             sizeof(cl_uint),
                             &foruzero,
                             0,
                             0,
                             0);

  CHECK_ERR(err);

  Mems_Worklist2 mwl1;
  Mems_Worklist2 mwl2;

  // Theoretically, the buffers should be size (hgraph.nedges * 2),
  // but some GPUs we tested do not have enough memory. Using less
  // memory works for all graphs and GPUs we've tested so far.
  cl_mem wl1 = init_worklist2(&context, &prog, &queue, hgraph->nedges * 1, &mwl1);
  cl_mem wl2 = init_worklist2(&context, &prog, &queue, hgraph->nedges * 1, &mwl2);

  // Create and initialise the unsafe global barrier
  cl_mem d_gbar_arr;
  err = allocate_and_init_gbar(&context, &queue, &d_gbar_arr,  wgn);
  CHECK_ERR(err);

  // Creating and setting args for the main kernel (drelax2)
  drelax2 = clCreateKernel(prog, "drelax2", &err);
  CHECK_ERR(err);

  err  = clSetKernelArg(drelax2, 0, sizeof(void *), (void*) dist);
  err  |= clSetKernelArg(drelax2, 1, sizeof(void *), (void*) dgraph);
  err  |= clSetKernelArg(drelax2, 2, sizeof(void *), (void*) &wl1);
  err  |= clSetKernelArg(drelax2, 3, sizeof(void *), (void*) &wl2);
  err  |= clSetKernelArg(drelax2, 4, sizeof(cl_int), (void*) &iteration);
  err  |= clSetKernelArg(drelax2, 5, sizeof(void *), (void*) &d_gbar_arr);
  CHECK_ERR(err);

  // Setting kernel dimensions
  size_t local_work[3] =  { kconf->wgs,  1, 1};
  size_t global_work[3] = { kconf->gs, 1,  1 };

  printf("solving.\n");
  printf("starting...\n");

  starttime = rtclock();

  // Launch the main kernel (drelax2). With iteration = 0, there's a
  // special path for some additional initialisation. We then need to
  // launch it again with iteration = 1.
  err = clEnqueueNDRangeKernel(queue,
                               drelax2,
                               1,
                               NULL,
                               global_work,
                               local_work,
                               0,
                               0,
                               0);

  CHECK_ERR(err);

  // Update dimensions for the non-init launch of drelax2
  local_work[0]  = kconf->wgs;
  global_work[0] = kconf->wgs * wgn;

  // Set iteration = 1 to trigger the non-init launch of drelax2
  iteration = 1;
  err  = clSetKernelArg(drelax2, 4, sizeof(cl_int), (void*) &iteration);
  CHECK_ERR(err);

  // Launch the main kernel (drelax2) with iteration = 1.
  err = clEnqueueNDRangeKernel(queue,
                               drelax2,
                               1,
                               NULL,
                               global_work,
                               local_work,
                               0,
                               0,
                               0);
  CHECK_ERR(err);
  err = clFinish(queue);
  CHECK_ERR(err);

  endtime = rtclock();

  // Print timing info
  printf("\tapp runtime = %f ms.\n", 1000.0f * (endtime - starttime));

  // Clean up device buffers
  dealloc_mems_wl(&mwl1);
  dealloc_mems_wl(&mwl2);
  clReleaseKernel(drelax2);
  clReleaseMemObject(wl1);
  clReleaseMemObject(wl2);
  clReleaseMemObject(d_gbar_arr);
  return;
}

// Main
int main(int argc, char *argv[]) {
  int err;
  foru *hdist;
  cl_mem dist, nerr, dgraph;
  cl_uint intzero = 0;
  Cl_Graph_mems graph_mems;
  cl_int *zero_array;
  Graph hgraph;
  unsigned hnerr;
  KernelConfig kconf;
  cl_kernel init, verify;

  // Parse and verify args
  if (argc != 4) {
    printf("Usage: %s <graph> <workgroup size> <workgroup number>\n", argv[0]);
    exit(1);
  }
  wgs = atoi(argv[2]);
  wgn = atoi(argv[3]);
  if (!pow_2(wgs)) {
    printf("this implementation requires a power of two workgroup size\n");
    exit(1);
  }

  // Init OpenCL, read the graph, and get the graph on the device side.
  init_opencl();
  read_graph(&hgraph, argv[1]);
  dgraph = copy_graph_gpu(&context, &prog, &queue, &hgraph, &graph_mems);

  // Create and initialise the distance graph on the GPU
  dist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_foru)*hgraph.nnodes, NULL ,&err);
  CHECK_ERR(err);
  hdist = (foru *) malloc(hgraph.nnodes * sizeof(cl_foru));

  // Create a variable to check for errors when we verify later
  nerr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL ,&err);
  CHECK_ERR(err);

  err = clEnqueueWriteBuffer(queue,
                             nerr,
                             1,
                             0,
                             sizeof(cl_uint),
                             &intzero,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  // Set kernel dimensions
  set_problem_size(&kconf, hgraph.nnodes);
  set_workgroup_size(&kconf, wgs);
  calculate_kernelconf(&kconf);

  // Create and set kernel args for the initialisation kernel
  init = clCreateKernel(prog, "initialize", &err);
  CHECK_ERR(err);

  err  = clSetKernelArg(init, 0, sizeof(void *), (void*) &dist);
  err |= clSetKernelArg(init, 1, sizeof(cl_int), &hgraph.nnodes);
  CHECK_ERR(err);

  size_t local_work[3]  = { kconf.wgs, 1, 1 };
  size_t global_work[3] = { kconf.gs,  1, 1 };

  printf("initializing.\n");

  // Launch initialisation kernel
  err = clEnqueueNDRangeKernel(queue,
                               init,
                               1,
                               NULL,
                               global_work,
                               local_work,
                               0,
                               0,
                               0);

  CHECK_ERR(err);
  err = clFinish(queue);
  CHECK_ERR(err);

  // Do sssp
  sssp(hdist, &dist, &hgraph, &dgraph, &kconf);

  // Now we verify the solution with the dverifysolution kernel
  printf("verifying.\n");

  verify = clCreateKernel(prog, "dverifysolution", &err);
  CHECK_ERR(err);

  err  = clSetKernelArg(verify, 0, sizeof(void *), (void*) &dist);
  err |= clSetKernelArg(verify, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(verify, 2, sizeof(void *), (void*) &nerr);
  CHECK_ERR(err);

  err = clEnqueueNDRangeKernel(queue,
                               verify,
                               1,
                               NULL,
                               global_work,
                               local_work,
                               0,
                               0,
                               0);
  CHECK_ERR(err);
  err = clFinish(queue);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(queue,
                            nerr,
                            1,
                            0,
                            sizeof(cl_int),
                            &hnerr,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  // Check that there were no errors
  printf("\tno of errors = %d.\n", hnerr);
  assert(hnerr == 0);

  clReleaseMemObject(nerr);
  clReleaseMemObject(dist);
  clReleaseMemObject(dgraph);
  clReleaseKernel(verify);
  clReleaseKernel(init);
  dealloc_cl_mems(&graph_mems);

  // Free host side memory
  free(hdist);

  // Clean-up OpenCL and print the device information
  clean_opencl();
  print_device_info();
  return 0;
}
