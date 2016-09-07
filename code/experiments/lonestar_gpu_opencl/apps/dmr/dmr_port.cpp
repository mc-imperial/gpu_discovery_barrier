/** Delaunay refinement -*- C++ -*-
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
 * Refinement of an initial, unrefined Delaunay mesh to eliminate triangles
 * with angles < 30 degrees
 *
 * @author: Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// The GPU-Lonestar dmr application ported to OpenCL.  Port by Tyler
// Sorensen (2016)

// This application uses the portable discovery protocol which
// requires no knowledge about the target GPU occupancy

#include <CL/cl.h>
#include "stdio.h"
#include "stdlib.h"
#include "assert.h"
#include "header.h"
#include <map>

#ifdef WIN32
typedef int uint;
#endif

// A high-level OpenCL check_err function
#define CHECK_ERR(err)  if (err < 0 ) { printf("CL_ERROR %s %d (%d)\n", __FILE__, __LINE__, err); exit(1); }

#include "worklistc.h"
#include "shmesh.h"
#include "meshfiles.h"
#include "my_opencl.h"
#include "discovery.h"

// OpenCL Utilitiesi
cl_context context;
cl_device_id device;
cl_command_queue queue;
cl_program prog;

int WGS; // Threads per workgroup
int WGN; // Number of workgroups

const char * CL_FILE = STRINGIFY(KERNEL_DIR) "dmr_port_kernel.cl";

#include "dmr.h"

void refine_mesh(ShMesh &mesh) {

  uint nbad, nelements, nnodes;
  int err, cnbad, zero = 0, lastnelements = 0;

  cl_mem d_nbad, d_mesh, d_nelements, d_nnodes, inwl, outwl;
  cl_mem d_gl_ctx;

  Mesh_mems mm;
  Mems_Worklist2 inmwl;
  Mems_Worklist2 outmwl;

  double starttime, endtime;

  cl_kernel check_triangles, refine;

  find_neighbours_cpu(mesh);

  // Create all of the device side buffers
  d_nelements = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
  CHECK_ERR(err);

  d_nnodes = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
  CHECK_ERR(err);

  d_nbad = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
  CHECK_ERR(err);

  // Create and initialise discovery protocol context
  d_gl_ctx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(discovery_kernel_ctx), NULL ,&err);
  err = init_discovery_kernel_ctx(&prog, &queue, &d_gl_ctx);
  CHECK_ERR(err);

  // Initialise some of the other device buffers
  err = clEnqueueWriteBuffer(queue,
                             d_nelements,
                             1,
                             0,
                             sizeof(cl_int),
                             &mesh.nelements,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  nelements = mesh.nelements;
  err = clEnqueueWriteBuffer(queue,
                             d_nnodes,
                             1,
                             0,
                             sizeof(cl_int),
                             &mesh.nnodes,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  nbad = 0;
  err = clEnqueueWriteBuffer(queue,
                             d_nbad,
                             1,
                             0,
                             sizeof(cl_int),
                             &nbad,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  // Create and move the mesh to the device
  d_mesh = create_and_init_device_mesh(&context, &queue, &prog, &mesh, &mm);
  nnodes = mesh.nnodes;

  // Create and initialise the worklists
  inwl = init_worklist2(&context, &prog, &queue, mesh.nelements, &inmwl);
  outwl = init_worklist2(&context, &prog, &queue, mesh.nelements, &outmwl);

  // Create the kernels
  check_triangles = clCreateKernel(prog, "check_triangles", &err);
  CHECK_ERR(err);

  refine = clCreateKernel(prog, "refine", &err);
  CHECK_ERR(err);

  // Set the kernel args
  err  = clSetKernelArg(check_triangles, 0, sizeof(void *), (void*) &d_mesh);
  err |= clSetKernelArg(check_triangles, 1, sizeof(void *), (void*) &d_nbad);
  err |= clSetKernelArg(check_triangles, 2, sizeof(void *), (void*) &inwl);
  err |= clSetKernelArg(check_triangles, 3, sizeof(cl_int), &zero);
  CHECK_ERR(err);

  err  = clSetKernelArg(refine, 0, sizeof(void *), (void*) &d_mesh);
  err |= clSetKernelArg(refine, 1, sizeof(void *), (void*) &d_nnodes);
  err |= clSetKernelArg(refine, 2, sizeof(void *), (void*) &d_nelements);
  err |= clSetKernelArg(refine, 3, sizeof(void *), (void*) &inwl);
  err |= clSetKernelArg(refine, 4, sizeof(void *), (void*) &outwl);
  err |= clSetKernelArg(refine, 5, sizeof(void *), (void*) &d_gl_ctx);
  CHECK_ERR(err);

  // Set kernel dimensions
  size_t global_size[3] = {WGN, 0, 0}, local_size[3] = {WGS, 0, 0};

  // Start computation
  starttime = rtclock();

  // First check triangles
  err = clEnqueueNDRangeKernel(queue,
                               check_triangles,
                               1,
                               NULL,
                               global_size,
                               local_size,
                               0,
                               0,
                               0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(queue,
                          d_nbad,
                          1,
                          0,
                          sizeof(cl_int),
                          &cnbad,
                          0,
                          0,
                          0);
  CHECK_ERR(err);

  int iteration = 0;

  // Loop while there exists bad triangles
  while (cnbad) {

    lastnelements = mesh.nelements;

    // Refine the mesh
    err = clEnqueueNDRangeKernel(queue,
                                 refine,
                                 1,
                                 NULL,
                                 global_size,
                                 local_size,
                                 0,
                                 0,
                                 0);
    CHECK_ERR(err);

    // Record the number of nodes and elements
    err = clEnqueueReadBuffer(queue,
                              d_nnodes,
                              1,
                              0,
                              sizeof(cl_int),
                              &nnodes,
                              0,
                              0,
                              0);
    CHECK_ERR(err);
    mesh.nnodes = nnodes;
    err = clEnqueueReadBuffer(queue,
                              d_nelements,
                              1,
                              0,
                              sizeof(cl_int),
                              &nelements,
                              0,
                              0,
                              0);

    CHECK_ERR(err);
    mesh.nelements = nelements;

    // Work around for OpenCL. In CUDA, the host can simply swap the
    // device pointers. Here we don't have device pointers, rather we
    // have the cl_mem objects. Thus, we just change cl_mem arg number
    // every other iteration
    if (iteration % 2 == 0) {
      err  = clSetKernelArg(refine, 4, sizeof(void *), (void*) &inwl);
      err |= clSetKernelArg(refine, 3, sizeof(void *), (void*) &outwl);
      err |= clSetKernelArg(check_triangles, 2, sizeof(void *), (void*) &outwl);
    }
    else {
      err  = clSetKernelArg(refine, 3, sizeof(void *), (void*) &inwl);
      err |= clSetKernelArg(refine, 4, sizeof(void *), (void*) &outwl);
      err |= clSetKernelArg(check_triangles, 2, sizeof(void *), (void*) &inwl);
    }
    CHECK_ERR(err);

    // We now have to reset one of the worklists depending on which
    // one we swapped
    if (iteration % 2 == 0) {

      // Reset inwl
      err = wl_reset(&queue, &inmwl);
    }
    else {

      // Reset outwl
      err = wl_reset(&queue, &outmwl);
    }
    CHECK_ERR(err);

    // Initialise to 0 bad triangles
    nbad = 0;
    err = clEnqueueWriteBuffer(queue,
                               d_nbad,
                               1,
                               0,
                               sizeof(cl_int),
                               &nbad,
                               0,
                               0,
                               0);

    CHECK_ERR(err);

    // Update the lastnelements arg (it changes in the loop)
    err = clSetKernelArg(check_triangles, 3, sizeof(cl_int), &lastnelements);
    CHECK_ERR(err);

    // Now check the triangles
    err = clEnqueueNDRangeKernel(queue,
                                 check_triangles,
                                 1,
                                 NULL,
                                 global_size,
                                 local_size,
                                 0,
                                 0,
                                 0);
    CHECK_ERR(err);

    // Again, based on the parity of the iteration, we
    // query the size of one of the worklists
    if (iteration % 2 == 0) {
      err = wl_get_nitems(&queue, &outmwl, &cnbad);
    }
    else {
      err = wl_get_nitems(&queue, &inmwl, &cnbad);
    }
    CHECK_ERR(err);

    // Break if there are no bad triangles
    if(cnbad == 0) {
      break;
    }
    iteration++;
  }

  err = clFinish(queue);
  CHECK_ERR(err);

  endtime = rtclock();

  // Print timing and runtime info
  printf("\tapp runtime = %f ms.\n", 1000.0f * (endtime - starttime));
  int participating_wgs = number_of_participating_groups(&queue, &d_gl_ctx);
  printf("\tnumber of participating groups = %d\n", participating_wgs);

  // Here we verify that there are no bad triangles in the mesh.  This
  // isn't enough for robust verification of the solution, but it is a
  // start. According to Sreepathi, the other property to check is if
  // the mesh has the delauny property.
  nbad = 0;
  err = clEnqueueWriteBuffer(queue,
                             d_nbad,
                             1,
                             0,
                             sizeof(cl_int),
                             &nbad,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  err = clEnqueueNDRangeKernel(queue,
                               check_triangles,
                               1,
                               NULL,
                               global_size,
                               local_size,
                               0,
                               0,
                               0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(queue,
                            d_nbad,
                            1,
                            0,
                            sizeof(cl_int),
                            &cnbad,
                            0,
                            0,
                            0);

  CHECK_ERR(err);

  cpy_mesh_to_host(&queue, &mm, &mesh);
  printf("%d final bad triangles\n", cnbad);
  assert(cnbad == 0);

  // Release kernels
  clReleaseKernel(check_triangles);
  clReleaseKernel(refine);

  // Release device buffers
  clReleaseMemObject(d_nbad);
  clReleaseMemObject(d_mesh);
  clReleaseMemObject(d_nelements);
  clReleaseMemObject(d_nnodes);
  clReleaseMemObject(d_gl_ctx);
  clReleaseMemObject(inwl);
  clReleaseMemObject(outwl);

  // Release the worklists and mesh device buffers
  dealloc_mems_wl(&inmwl);
  dealloc_mems_wl(&outmwl);
  free_mesh_mems(&mm);
}

// Main
int main(int argc, char *argv[]) {
  ShMesh mesh;
  int maxfactor = 2;
  int mesh_nodes, mesh_elements;

  // Check and parse command line args
  if(argc != 4) {
      printf("Usage: %s basefile <workgroup size> <maxfactor>\n", argv[0]);
      exit(0);
    }

  maxfactor = atoi(argv[3]);
  WGS = atoi(argv[2]);
  WGN = WGS*1000;

  // Read in the mesh and intiialise some local variables
  read_mesh(argv[1], mesh, maxfactor);
  mesh_nodes = mesh.nnodes;
  mesh_elements = mesh.ntriangles + mesh.nsegments;

  // Intialise OpenCL utilities
  init_opencl();

  // Do dmr
  refine_mesh(mesh);

  // Print some information about the solution
  printf("%f increase in number of elements (maxfactor hint)\n", 1.0 * mesh.nelements / mesh_elements);
  printf("%f increase in number of nodes (maxfactor hint)\n", 1.0 * mesh.nnodes / mesh_nodes);

  // Free the mesh
  free_mesh(&mesh);

  // Clean up the OpenCL utilities and print device information
  clean_opencl();
  print_device_info();

  return 0;
}
