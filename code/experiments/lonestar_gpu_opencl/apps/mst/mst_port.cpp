/** Minimum spanning tree -*- C++ -*-
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
 * @Description
 * Computes minimum spanning tree of a graph using Boruvka's algorithm.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// The GPU-Lonestar mst application ported to OpenCL.  Port by
// Tyler Sorensen (2016)

// This application uses the portable discovery protocol which
// requires no knowledge about the target GPU occupancy

typedef unsigned foru;

#include <CL/cl.h>
#include "stdio.h"
#include "common.h"
#include "header.h"
#include "graph.h"
#include "component.h"
#include "kernelconf.h"
#include "my_opencl.h"
#include "discovery.h"
#include "assert.h"

const char * CL_FILE = STRINGIFY(KERNEL_DIR) "mst_port_kernel.cl";

// Device side buffers for mst.
cl_mem mstwt, eleminwts, minwtcomponent, partners,
  phores, processinnextiteration, goaheadnodeofcomponent, grepeat, gedgecount;

#include "mst.h"

// Main
int main(int argc, char *argv[]) {

  // OpenCL utility variables
  cl_context context;
  cl_device_id device;
  cl_command_queue queue;

  int err = 0;
  int iteration = 0;
  int repeat = 0;
  unsigned hmstwt = 0;
  int awgs;

  Graph hgraph;
  KernelConfig kconf;

  Cl_Graph_mems graph_mems;
  ComponentSpace_mems cs_mems;

  double starttime, endtime;

  // Check command line args
  if (argc != 3) {
    printf("Usage: %s <graph> <workgroup_size>\n", argv[0]);
    return 1;
  }

  awgs = atoi(argv[2]);

  // Initisatise OpenCL utilities
  device = create_device();

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err < 0 ) { perror("Couldn't create OpenCL context"); exit(1); }

  queue = clCreateCommandQueue(context, device, 0, &err);
  if (err < 0 ) { perror("failed create command queue barrier"); exit(1); }

  // Compile the kernel file
  char opts[500];
  get_compile_opts(opts);
  cl_program prog = build_program(context, device, CL_FILE, opts);

  // Read the graph and copy it to device side buffers
  read_graph(&hgraph, argv[1]);
  cl_mem dgraph = copy_graph_gpu(&context, &prog, &queue, &hgraph, &graph_mems);
  allocate_cl_mems(&context, &queue, hgraph);

  // Get device side component space
  cl_mem cs = create_and_init_cs(&context, &prog, &queue, &cs_mems, hgraph.nnodes);

  // Create and initialise discovery protocol
  cl_mem d_gl_ctx;
  d_gl_ctx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(discovery_kernel_ctx), NULL ,&err);
  err = init_discovery_kernel_ctx(&prog, &queue, &d_gl_ctx);
  if (err < 0 ) { perror("failed initialising discovery kernel context"); exit(1); }

  // Set kernel dimensions
  unsigned prevncomponents, currncomponents = hgraph.nnodes;
  unsigned edgecount = 0;
  set_problem_size(&kconf, hgraph.nnodes);
  set_workgroup_size(&kconf, 128);
  calculate_kernelconf(&kconf);

  size_t local_work[3]  = { kconf.wgs, 1, 1 };
  size_t global_work[3] = { kconf.gs,  1, 1 };

  // Create kernels and set args
  cl_kernel dinit = clCreateKernel(prog, "dinit", &err);
  if (err < 0 ) { printf("failed getting a kernel %d\n", err); exit(1); }

  err  = clSetKernelArg(dinit, 0, sizeof(void *), (void*) &mstwt);
  err |= clSetKernelArg(dinit, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(dinit, 2, sizeof(void *), (void *) &cs);
  err |= clSetKernelArg(dinit, 3, sizeof(void *), (void *) &eleminwts);
  err |= clSetKernelArg(dinit, 4, sizeof(void *), (void *) &minwtcomponent);
  err |= clSetKernelArg(dinit, 5, sizeof(void *), (void *) &partners);
  err |= clSetKernelArg(dinit, 6, sizeof(void *), (void *) &phores);
  err |= clSetKernelArg(dinit, 7, sizeof(void *), (void *) &processinnextiteration);
  err |= clSetKernelArg(dinit, 8, sizeof(void *), (void *) &goaheadnodeofcomponent);
  if (err < 0 ) { printf("failed setting kernel arg dinit %d\n", err); exit(1); }

  cl_kernel dfindelemin = clCreateKernel(prog, "dfindelemin", &err);
  if (err < 0 ) { printf("failed getting a kernel %d\n", err); exit(1); }

  err  = clSetKernelArg(dfindelemin, 0, sizeof(void *), (void*) &mstwt);
  err |= clSetKernelArg(dfindelemin, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(dfindelemin, 2, sizeof(void *), (void *) &cs);
  err |= clSetKernelArg(dfindelemin, 3, sizeof(void *), (void *) &eleminwts);
  err |= clSetKernelArg(dfindelemin, 4, sizeof(void *), (void *) &minwtcomponent);
  err |= clSetKernelArg(dfindelemin, 5, sizeof(void *), (void *) &partners);
  err |= clSetKernelArg(dfindelemin, 6, sizeof(void *), (void *) &phores);
  err |= clSetKernelArg(dfindelemin, 7, sizeof(void *), (void *) &processinnextiteration);
  err |= clSetKernelArg(dfindelemin, 8, sizeof(void *), (void *) &goaheadnodeofcomponent);
  if (err < 0 ) { printf("failed setting kernel arg dfindelemin %d\n", err); exit(1); }

  cl_kernel dfindelemin2 = clCreateKernel(prog, "dfindelemin2", &err);
  if (err < 0 ) { printf("failed getting a kernel %d\n", err); exit(1); }

  err  = clSetKernelArg(dfindelemin2, 0, sizeof(void *), (void*) &mstwt);
  err |= clSetKernelArg(dfindelemin2, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(dfindelemin2, 2, sizeof(void *), (void *) &cs);
  err |= clSetKernelArg(dfindelemin2, 3, sizeof(void *), (void *) &eleminwts);
  err |= clSetKernelArg(dfindelemin2, 4, sizeof(void *), (void *) &minwtcomponent);
  err |= clSetKernelArg(dfindelemin2, 5, sizeof(void *), (void *) &partners);
  err |= clSetKernelArg(dfindelemin2, 6, sizeof(void *), (void *) &phores);
  err |= clSetKernelArg(dfindelemin2, 7, sizeof(void *), (void *) &processinnextiteration);
  err |= clSetKernelArg(dfindelemin2, 8, sizeof(void *), (void *) &goaheadnodeofcomponent);
  if (err < 0 ) { printf("failed setting kernel arg dfindelemin2 %d\n", err); exit(1); }

  cl_kernel verify_min_elem = clCreateKernel(prog, "verify_min_elem", &err);
  if (err < 0 ) { printf("failed getting a kernel %d\n", err); exit(1); }

  err  = clSetKernelArg(verify_min_elem, 0, sizeof(void *), (void*) &mstwt);
  err |= clSetKernelArg(verify_min_elem, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(verify_min_elem, 2, sizeof(void *), (void *) &cs);
  err |= clSetKernelArg(verify_min_elem, 3, sizeof(void *), (void *) &eleminwts);
  err |= clSetKernelArg(verify_min_elem, 4, sizeof(void *), (void *) &minwtcomponent);
  err |= clSetKernelArg(verify_min_elem, 5, sizeof(void *), (void *) &partners);
  err |= clSetKernelArg(verify_min_elem, 6, sizeof(void *), (void *) &phores);
  err |= clSetKernelArg(verify_min_elem, 7, sizeof(void *), (void *) &processinnextiteration);
  err |= clSetKernelArg(verify_min_elem, 8, sizeof(void *), (void *) &goaheadnodeofcomponent);
  if (err < 0 ) { printf("failed setting kernel arg verify_min_elem %d\n", err); exit(1); }

  cl_kernel dfindcompmintwo = clCreateKernel(prog, "dfindcompmintwo", &err);
  if (err < 0 ) { printf("failed getting a kernel %d\n", err); exit(1); }

  err  = clSetKernelArg(dfindcompmintwo, 0, sizeof(void *), (void*) &mstwt);
  err |= clSetKernelArg(dfindcompmintwo, 1, sizeof(void *), (void*) &dgraph);
  err |= clSetKernelArg(dfindcompmintwo, 2, sizeof(void *), (void *) &cs);
  err |= clSetKernelArg(dfindcompmintwo, 3, sizeof(void *), (void *) &eleminwts);
  err |= clSetKernelArg(dfindcompmintwo, 4, sizeof(void *), (void *) &minwtcomponent);
  err |= clSetKernelArg(dfindcompmintwo, 5, sizeof(void *), (void *) &partners);
  err |= clSetKernelArg(dfindcompmintwo, 6, sizeof(void *), (void *) &phores);
  err |= clSetKernelArg(dfindcompmintwo, 7, sizeof(void *), (void *) &processinnextiteration);
  err |= clSetKernelArg(dfindcompmintwo, 8, sizeof(void *), (void *) &goaheadnodeofcomponent);
  err |= clSetKernelArg(dfindcompmintwo, 9, sizeof(void *), (void *) &grepeat);
  err |= clSetKernelArg(dfindcompmintwo, 10, sizeof(void *), (void *) &gedgecount);
  err |= clSetKernelArg(dfindcompmintwo, 11, sizeof(void *), (void *) &d_gl_ctx);
  if (err < 0 ) { printf("failed setting kernel arg dfindcompmintwo %d\n", err); exit(1); }

  size_t local_work_active[3] =  { awgs,  1, 1};
  size_t global_work_active[3] = { 1000 * awgs, 1,  1 };

  // Start kernel computations
  printf("finding mst.\n");
  starttime = rtclock();
  do {

    iteration++;
    prevncomponents = currncomponents;

    // Launch the init kernel
    err = clEnqueueNDRangeKernel(queue,
                                 dinit,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: dinit (%d)\n", err); return -1; }

    // Launch dfindelemin
    err = clEnqueueNDRangeKernel(queue,
                                 dfindelemin,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: dfindelemin (%d)\n", err); return -1; }

    // Launch dfindelemin2
    err = clEnqueueNDRangeKernel(queue,
                                 dfindelemin2,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: dfindelemin2 (%d)\n", err); return -1; }

    // Launch verify_min_elem
    err = clEnqueueNDRangeKernel(queue,
                                 verify_min_elem,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: verify_min_elem (%d)\n", err); return -1; }

    // Inner computation loop
    do {

      // Assume we do not repeat
      repeat = 0;
      err = clEnqueueWriteBuffer(queue,
                                 grepeat,
                                 1,
                                 0,
                                 sizeof(cl_int),
                                 &repeat,
                                 0,
                                 0,
                                 0);
      if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying memory to device %d\n", err); return -1; }

      // Launch dfindcompmintwo
      err = clEnqueueNDRangeKernel(queue,
                                   dfindcompmintwo,
                                   1,
                                   NULL,
                                   global_work_active,
                                   local_work_active,
                                   0,
                                   0,
                                   0);
      if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: dfindcompmintwo (%d)\n", err); return -1; }

      // Check to see if we have to repeat
      err = clEnqueueReadBuffer(queue,
                                grepeat,
                                1,
                                0,
                                sizeof(cl_int),
                                &repeat,
                                0,
                                0,
                                0);
      if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying memory from device %d\n", err); return -1; }

    } while (repeat);

    // Check the number of components
    err = clEnqueueReadBuffer(queue,
                              cs_mems.ncomponents,
                              1,
                              0,
                              sizeof(cl_int),
                              &currncomponents,
                              0,
                              0,
                              0);
    if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying memory from device %d\n", err); return -1; }


    // Loop as long as we're finding new components
  } while (currncomponents != prevncomponents);

  err = clFinish(queue);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: waiting for queues to finish (%d)\n", err); return -1; }

  endtime = rtclock();

  // Read back the weight and edgecount to verify
  err = clEnqueueReadBuffer(queue,
                            mstwt,
                            1,
                            0,
                            sizeof(cl_int),
                            &hmstwt,
                            0,
                            0,
                            0);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying memory from device %d\n", err); return -1; }

  err = clEnqueueReadBuffer(queue,
                            gedgecount,
                            1,
                            0,
                            sizeof(cl_int),
                            &edgecount,
                            0,
                            0,
                            0);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying memory from device %d\n", err); return -1; }

  // Verify the solution
  check_outcome(argv[1], hmstwt, currncomponents, edgecount);

  // Print timing, runtime and solution information
  printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[1], hmstwt, currncomponents, edgecount);
  printf("\tapp runtime = %f ms.\n", 1000 * (endtime - starttime));

  int participating_wgs = number_of_participating_groups(&queue, &d_gl_ctx);
  printf("\tnumber of participating groups = %d\n", participating_wgs);

  // Free up memory and print device info
  free_host_graph(&hgraph);
  dealloc_cl_mems(&graph_mems);
  release_cl_mems();
  print_device_info();

  return 0;
}
