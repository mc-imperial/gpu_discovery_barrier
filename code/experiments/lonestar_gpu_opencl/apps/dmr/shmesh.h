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

// The GPU-Lonestar sssp-dmr application ported to OpenCL.  Port by
// Tyler Sorensen (2016)

#pragma once

typedef struct {
  uint x;
  uint y;
  uint z;
} uint3;

#define MINANGLE 30
#define PI 3.14159265358979323846 // From C99 standard.
#define FORD double
#define DIMSTYPE unsigned

#define INVALIDID 1234567890
#define MAXID INVALIDID

#define MAX_NNODES_TO_NELEMENTS 2

typedef struct  {
  uint maxnelements;
  uint maxnnodes;
  uint ntriangles;
  uint nnodes;
  uint nsegments;
  uint nelements;

  FORD * nodex;
  FORD * nodey;
  cl_uint3 * elements;
  cl_uint3 * neighbours;
  int * isdel;
  int * isbad;
  int * owners;
} ShMesh;

typedef struct {
  cl_uint maxnelements;
  cl_uint maxnnodes;
  cl_uint ntriangles;
  cl_uint nnodes;
  cl_uint nsegments;
  cl_uint nelements;

  FORD *nodex;
  FORD *nodey;
  cl_uint3 *elements;
  cl_int *isdel;
  cl_int *isbad;
  cl_uint3 *neighbours;
  cl_int *owners;
} Mesh_cl;

typedef struct {
  cl_mem nodex;
  cl_mem nodey;
  cl_mem elements;
  cl_mem isdel;
  cl_mem isbad;
  cl_mem neighbours;
  cl_mem owners;
} Mesh_mems;

void allocate_mesh_mems(cl_context *c, cl_command_queue *q, cl_program *p, ShMesh *shm, Mesh_mems *mm) {

  int err;
  mm->nodex = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(FORD) * shm->maxnnodes, NULL, &err);
  CHECK_ERR(err);

  mm->nodey = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(FORD)*shm->maxnnodes, NULL, &err);
  CHECK_ERR(err);

  mm->elements = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint3) * shm->maxnelements, NULL, &err);
  CHECK_ERR(err);


  mm->isdel = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_int) * shm->maxnelements, NULL, &err);
  CHECK_ERR(err);

  mm->isbad = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_int) * shm->maxnelements, NULL, &err);
  CHECK_ERR(err);

  mm->neighbours = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint3) * shm->maxnelements, NULL, &err);
  CHECK_ERR(err);

  mm->owners = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_int) * shm->maxnelements, NULL, &err);
  CHECK_ERR(err);
}

void copy_mesh_mems(cl_context *c, cl_command_queue *q, cl_program *p, ShMesh *shm, Mesh_mems *mm) {
  int err;
  err = clEnqueueWriteBuffer(*q,
                             mm->nodex,
                             1,
                             0,
                             sizeof(FORD) * shm->maxnnodes,
                             shm->nodex,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  err = clEnqueueWriteBuffer(*q,
                             mm->nodey,
                             1,
                             0,
                             sizeof(FORD) * shm->maxnnodes,
                             shm->nodey,
                             0,
                             0,
                             0);
  CHECK_ERR(err);


  err = clEnqueueWriteBuffer(*q,
                             mm->elements,
                             1,
                             0,
                             sizeof(cl_uint3) * shm->maxnelements,
                             shm->elements,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  err = clEnqueueWriteBuffer(*q,
                             mm->isdel,
                             1,
                             0,
                             sizeof(cl_int) * shm->maxnelements,
                             shm->isdel,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  err = clEnqueueWriteBuffer(*q,
                             mm->isbad,
                             1,
                             0,
                             sizeof(cl_int) * shm->maxnelements,
                             shm->isbad,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  err = clEnqueueWriteBuffer(*q,
                             mm->neighbours,
                             1,
                             0,
                             sizeof(cl_uint3) * shm->maxnelements,
                             shm->neighbours,
                             0,
                             0,
                             0);
  CHECK_ERR(err);

  err = clEnqueueWriteBuffer(*q,
                             mm->owners,
                             1,
                             0,
                             sizeof(cl_int) * shm->maxnelements,
                             shm->owners,
                             0,
                             0,
                             0);
  CHECK_ERR(err);
}

void initialize_mesh_struct(cl_context *c, cl_command_queue *q, cl_program *p, ShMesh *shm, Mesh_mems *mm, cl_mem *ret) {
  int err;
  cl_kernel kernel;

  kernel = clCreateKernel(*p, "init_mesh", &err);
  CHECK_ERR(err);

  err  = clSetKernelArg(kernel, 0, sizeof(void *), (void*) ret);
  err |= clSetKernelArg(kernel, 1, sizeof(void *), (void*) &(mm->nodex));
  err |= clSetKernelArg(kernel, 2, sizeof(void *), (void*) &(mm->nodey));
  err |= clSetKernelArg(kernel, 3, sizeof(void *), (void*) &(mm->elements));
  err |= clSetKernelArg(kernel, 4, sizeof(void *), (void*) &(mm->neighbours));
  err |= clSetKernelArg(kernel, 5, sizeof(void *), (void*) &(mm->isdel));
  err |= clSetKernelArg(kernel, 6, sizeof(void *), (void*) &(mm->isbad));
  err |= clSetKernelArg(kernel, 7, sizeof(void *), (void*) &(mm->owners));

  err |= clSetKernelArg(kernel, 8, sizeof(cl_uint), (void*) &(shm->maxnelements));
  err |= clSetKernelArg(kernel, 9, sizeof(cl_uint), (void*) &(shm->maxnnodes));
  err |= clSetKernelArg(kernel, 10, sizeof(cl_uint), (void*) &(shm->ntriangles));
  err |= clSetKernelArg(kernel, 11, sizeof(cl_uint), (void*) &(shm->nnodes));
  err |= clSetKernelArg(kernel, 12, sizeof(cl_uint), (void*) &(shm->nsegments));
  err |= clSetKernelArg(kernel, 13, sizeof(cl_uint), (void*) &(shm->nelements));

  CHECK_ERR(err);

  size_t global_size[3] = {1, 0, 0}, local_size[3] = {1, 0, 0};

  err = clEnqueueNDRangeKernel(*q,
                               kernel,
                               1,
                               NULL,
                               global_size,
                               local_size,
                               0,
                               0,
                               0);

  CHECK_ERR(err);

  err = clFinish(*q);
  CHECK_ERR(err);
  clReleaseKernel(kernel);
}

cl_mem create_and_init_device_mesh(cl_context *c, cl_command_queue *q, cl_program *p, ShMesh *shm, Mesh_mems *mm) {

  int err;
  cl_mem ret = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(Mesh_cl), NULL, &err);
  CHECK_ERR(err);
  allocate_mesh_mems(c, q, p, shm, mm);
  copy_mesh_mems(c, q, p, shm, mm);

  initialize_mesh_struct(c, q, p, shm, mm, &ret);

  return ret;
}

void cpy_mesh_to_host(cl_command_queue *q, Mesh_mems *mm, ShMesh *shm) {

  int err;

  err = clEnqueueReadBuffer(*q,
                            mm->nodex,
                            1,
                            0,
                            sizeof(FORD) * shm->maxnnodes,
                            shm->nodex,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(*q,
                            mm->nodey,
                            1,
                            0,
                            sizeof(FORD) * shm->maxnnodes,
                            shm->nodey,
                            0,
                            0,
                            0);
  CHECK_ERR(err);


  err = clEnqueueReadBuffer(*q,
                            mm->elements,
                            1,
                            0,
                            sizeof(cl_uint3) * shm->maxnelements,
                            shm->elements,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(*q,
                            mm->isdel,
                            1,
                            0,
                            sizeof(cl_int) * shm->maxnelements,
                            shm->isdel,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(*q,
                            mm->isbad,
                            1,
                            0,
                            sizeof(cl_int) * shm->maxnelements,
                            shm->isbad,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(*q,
                            mm->neighbours,
                            1,
                            0,
                            sizeof(cl_uint3) * shm->maxnelements,
                            shm->neighbours,
                            0,
                            0,
                            0);
  CHECK_ERR(err);

  err = clEnqueueReadBuffer(*q,
                            mm->owners,
                            1,
                            0,
                            sizeof(cl_int) * shm->maxnelements,
                            shm->owners,
                            0,
                            0,
                            0);
  CHECK_ERR(err);


}

void free_mesh_mems(Mesh_mems *mm) {

  clReleaseMemObject(mm->nodex);
  clReleaseMemObject(mm->nodey);
  clReleaseMemObject(mm->elements);
  clReleaseMemObject(mm->isdel);
  clReleaseMemObject(mm->isbad);
  clReleaseMemObject(mm->neighbours);
  clReleaseMemObject(mm->owners);
}

#define IS_SEGMENT(element) (((element).z == INVALIDID))
