// Header file to use on the host for using the discovery protocol.
// Intialises the discovery protocol, barrier, mutexes, etc.

#pragma once

#include "common.h"

// This function initialises the discovery_kernel_ctx object for regular use.
// Should be called before a kernel using the discovery protocol.
int init_discovery_kernel_ctx(cl_program *p, cl_command_queue *q, cl_mem *gl_ctx) {
  cl_kernel kernel;
  int err;
  cl_int skip = 0;

  kernel = clCreateKernel(*p, "init_discovery_kernel_ctx",&err);
  if (err < 0 ) { return err; }

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), gl_ctx);
  if (err < 0 ) {  return err; }

  err = clSetKernelArg(kernel, 1, sizeof(cl_int), &skip);
  if (err < 0 ) {  return err; }

  size_t global_size[3] = {1, 0, 0}, local_size[3] = {1, 0, 0};
  err = clEnqueueNDRangeKernel(*q, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
  if (err < 0 ) { return err; }

  clFinish(*q);
  clReleaseKernel(kernel);

  return CL_SUCCESS;
}

// This function initialises the discovery_kernel_ctx object to skip the protocol.
// This is used to find the occupancy bound N in the occupancy tests.
int init_discovery_kernel_ctx_skip(cl_program *p, cl_command_queue *q, cl_mem *gl_ctx, int skip_arg) {
  cl_kernel kernel;
  int err;
  cl_int skip = skip_arg;

  kernel = clCreateKernel(*p, "init_discovery_kernel_ctx",&err);
  if (err < 0) { return err; }

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), gl_ctx);
  if (err < 0 ) { return err; }

  err = clSetKernelArg(kernel, 1, sizeof(cl_int), &skip);
  if (err < 0 ) { return err; }


  size_t global_size[3] = {1, 0, 0}, local_size[3] = {1, 0, 0};
  err = clEnqueueNDRangeKernel(*q, kernel, 1, NULL, global_size, local_size, 0, NULL, NULL);
  if (err < 0 ) {  return err; }

  clFinish(*q);
  clReleaseKernel(kernel);

  return CL_SUCCESS;
}

// After a kernel has run using the discovery protocol, this reports how many
// groups were discovered and determined to be participating.
int number_of_participating_groups(cl_command_queue *queue, cl_mem *gl_ctx) {
  discovery_kernel_ctx h_gl_ctx;
  int err;
  err = clEnqueueReadBuffer(*queue, *gl_ctx, CL_TRUE, 0, sizeof(discovery_kernel_ctx), &h_gl_ctx, 0, NULL, NULL);
  return h_gl_ctx.num_participating;
}

// Code used in the occupancy_test experiments to time the discovery protocol
int time_protocol(cl_program *p, cl_command_queue *q, cl_kernel *k, int upper_bound, int wgs, cl_mem *gl_ctx, double *time) {
  int err;

  err = init_discovery_kernel_ctx(p, q, gl_ctx);
  if (err < 0) { return err; }


  const size_t global_size = upper_bound * wgs;
  const size_t local_size = wgs;
  cl_event event;

  clFinish(*q);

  err = clEnqueueNDRangeKernel(*q,
                               *k,
                               1,
                               NULL,
                               &global_size,
                               &local_size,
                               0,
                               0,
                               &event);

  if (err < 0) { return err; }
  err = clWaitForEvents(1 , &event);
  if (err < 0) { return err; }

  cl_ulong time_start, time_end;

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  *time = time_end - time_start;

  return CL_SUCCESS;
}
