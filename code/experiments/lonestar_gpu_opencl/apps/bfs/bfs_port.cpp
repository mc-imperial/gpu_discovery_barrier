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

// The GPU-Lonestar bfs-wlc application ported to OpenCL.  Port by
// Tyler Sorensen (2016)

// This application uses the portable discovery protocol which
// requires no knowledge about the target GPU occupancy

// A high-level OpenCL check_err function
#define CHECK_ERR(err)  if (err < 0 ) { printf("CL_ERROR %s %d (%d)\n", __FILE__, __LINE__, err); exit(1); }

typedef unsigned foru;

#include "my_opencl.h"
#include "discovery.h"
#include "common.h"
#include "header.h"
#include "graph.h"
#include "kernelconf.h"
#include "worklistc.h"

typedef cl_uint cl_foru;

// OpenCL utilities
cl_context context;
cl_device_id device;
cl_command_queue queue;
cl_program prog;
int wgs;

const char * CL_FILE = STRINGIFY(KERNEL_DIR) "bfs_port_kernel.cl";

#include "bfs.h"

// Top level bfs function
void bfs(Graph hgraph, cl_mem *dist, cl_mem *dgraph) {
  cl_foru foru_zero = 0;
  int err;
  foru foruzero = 0;
  unsigned int NBLOCKS, FACTOR;
  int iteration = 0;

  double starttime, endtime;
  double runtime;
  int NVERTICES;
  cl_kernel init;
  cl_kernel drelax2;

  NVERTICES = hgraph.nnodes;

  // In the OpenCL port, MAXBLOCKSIZE is just the number of threads
  // per workgroups
  int MAXBLOCKSIZE = wgs;
  NBLOCKS = 1000;
  FACTOR = (NVERTICES + MAXBLOCKSIZE * NBLOCKS - 1) / (MAXBLOCKSIZE * NBLOCKS);

  // Creating kernels
  init = clCreateKernel(prog, "initialize", &err);
  CHECK_ERR(err);
  drelax2 = clCreateKernel(prog, "drelax2", &err);
  CHECK_ERR(err);

  // Setting initialisation args
  err  = clSetKernelArg(init, 0, sizeof(void *), (void*) dist);
  err |= clSetKernelArg(init, 1, sizeof(cl_int), &NVERTICES);
  CHECK_ERR(err);

  // Launching init kernel
  printf("initializing (nblocks=%d, blocksize=%d).\n", NBLOCKS*FACTOR, MAXBLOCKSIZE);
  printf("analyzing %d nodes\n", NVERTICES);

  size_t local_work[3]  = { MAXBLOCKSIZE,  1, 1 };
  size_t global_work[3] = { MAXBLOCKSIZE*NBLOCKS*FACTOR, 1, 1 };

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

  err = clEnqueueWriteBuffer(queue,
                             *dist,
                             1,
                             0,
                             sizeof(cl_foru),
                             &foru_zero,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  // Creating buffers for bfs
  cl_mem changed = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
  CHECK_ERR(err);
  cl_mem nerr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
  CHECK_ERR(err);

  Mems_Worklist2 mwl1;
  Mems_Worklist2 mwl2;

  // Theoretically, the buffers should be size (hgraph.nedges * 2),
  // but some GPUs we tested do not have enough memory. Using less
  // memory works for all graphs and GPUs we've tested so far.
  cl_mem wl1 = init_worklist2(&context, &prog, &queue, hgraph.nedges * 1, &mwl1);
  cl_mem wl2 = init_worklist2(&context, &prog, &queue, hgraph.nedges * 1, &mwl2);

  // Discovery protocol init
  cl_mem d_gl_ctx;
  d_gl_ctx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(discovery_kernel_ctx), NULL, &err);
  err = init_discovery_kernel_ctx(&prog, &queue, &d_gl_ctx);
  CHECK_ERR(err);

  // Setting args for the main kernel (drelax2)
  err   = clSetKernelArg(drelax2, 0, sizeof(void *), (void*) dist);
  err  |= clSetKernelArg(drelax2, 1, sizeof(void *), (void*) dgraph);
  err  |= clSetKernelArg(drelax2, 2, sizeof(void *), (void*) &nerr);
  err  |= clSetKernelArg(drelax2, 3, sizeof(void *), (void*) &wl1);
  err  |= clSetKernelArg(drelax2, 4, sizeof(void *), (void*) &wl2);
  err  |= clSetKernelArg(drelax2, 5, sizeof(cl_int), (void*) &iteration);
  err  |= clSetKernelArg(drelax2, 6, sizeof(void *), (void*) &d_gl_ctx);
  CHECK_ERR(err);

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
  local_work[0] =  wgs;
  global_work[0] = wgs*1000;

  // Set iteration = 1 to trigger the non-init launch of drelax2
  iteration = 1;
  err = clSetKernelArg(drelax2, 5, sizeof(cl_int), (void*) &iteration);
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

  // Print timing and runtime info
  printf("\tapp runtime = %f ms.\n", 1000.0f * (endtime - starttime));
  int participating_wgs = number_of_participating_groups(&queue, &d_gl_ctx);
  printf("\tnumber of participating groups = %d\n", participating_wgs);

  // Clean up device buffers
  clReleaseMemObject(changed);
  clReleaseMemObject(nerr);
  clReleaseMemObject(d_gl_ctx);
  clReleaseMemObject(wl1);
  clReleaseMemObject(wl2);
  clReleaseKernel(init);
  clReleaseKernel(drelax2);

  // Clean up host buffers
  dealloc_mems_wl(&mwl1);
  dealloc_mems_wl(&mwl2);

  return;
}


// Main
int main(int argc, char *argv[]) {
  int err;
  cl_uint intzero = 0;
  Graph hgraph;
  cl_mem dist, nerr;
  Cl_Graph_mems graph_mems;
  cl_int *zero_array;
  KernelConfig kconf;
  cl_kernel verify;

  // Parse and verify args
  if (argc != 3) {
    printf("Usage: %s <graph> <workgroup size>\n", argv[0]);
    exit(1);
  }
  wgs = atoi(argv[2]);
  if (!pow_2(wgs)) {
    printf("this implementation requires a power of two workgroup size\n");
    exit(1);
  }

  // Init OpenCL and read the graph
  init_opencl();
  read_graph(&hgraph, argv[1]);

  // Create and initialise the distance graph on the GPU
  dist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_foru)*hgraph.nnodes, NULL, &err);
  CHECK_ERR(err);

  zero_array = (cl_int*) malloc(hgraph.nnodes * sizeof(cl_int));
  for (int i = 0; i < hgraph.nnodes; i++) {
    zero_array[i] = 0;
  }

  err = clEnqueueWriteBuffer(queue,
                             dist,
                             1,
                             0,
                             sizeof(cl_int) * hgraph.nnodes,
                             zero_array,
                             0,
                             0,
                             0);
  CHECK_ERR(err);


  // Get the graph on the GPU
  cl_mem dgraph = copy_graph_gpu(&context, &prog, &queue, &hgraph, &graph_mems);

  // Do the bfs
  bfs(hgraph, &dist, &dgraph);

  // Verify the solution using the kernel dverifysolution
  printf("verifying.\n");
  nerr = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
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

  set_problem_size(&kconf, hgraph.nnodes);

  set_workgroup_size(&kconf, 128);
  calculate_kernelconf(&kconf);
  verify = clCreateKernel(prog, "dverifysolution", &err);
  CHECK_ERR(err);

  err  = clSetKernelArg(verify, 0, sizeof(void *), (void*) &dist);
  err |= clSetKernelArg(verify, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(verify, 2, sizeof(void *), (void *) &nerr);

  size_t local_work[3] =  { kconf.wgs,  1, 1};
  size_t global_work[3] = { kconf.gs, 1,  1 };

  printf("running verify with local size %d global size %d\n", kconf.wgs, kconf.gs);

  err = clEnqueueNDRangeKernel(queue,
                               verify,
                               1,
                               NULL,
                               global_work,
                               local_work,
                               0,
                               0,
                               0);
  err = clFinish(queue);
  CHECK_ERR(err);

  cl_uint hnerr;
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

  printf("\tno of errors = %d.\n", hnerr);
  assert(hnerr == 0);

  // Free graph and verification buffers
  clReleaseMemObject(nerr);
  clReleaseMemObject(dist);
  clReleaseMemObject(dgraph);
  clReleaseKernel(verify);
  dealloc_cl_mems(&graph_mems);

  // Free host side memory
  free_host_graph(&hgraph);
  free(zero_array);

  // Clean-up OpenCL and print the device information
  clean_opencl();
  print_device_info();

  return 0;
}
