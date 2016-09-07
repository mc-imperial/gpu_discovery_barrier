// The host part of the component space for the mst GPU-Lonestar
// application

// OpenCL port by Tyler Sorensen (2016)

#pragma once

typedef struct {
  cl_uint *ncomponents, *complen, *ele2comp;
} ComponentSpace;

typedef struct {
  cl_mem ncomponents, complen, ele2comp;
} ComponentSpace_mems;

void allocate_csm(cl_context *c, ComponentSpace_mems *csm, unsigned nelements) {
  int err;
  csm->ncomponents = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating ComponentSpace on device clCreateBuffer => %d\n", err); exit(1);}

  csm->complen = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint)* nelements, NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating ComponentSpace on device clCreateBuffer => %d\n", err); exit(1);}

  csm->ele2comp = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint)* nelements, NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating ComponentSpace on device clCreateBuffer => %d\n", err); exit(1);}
}

void deallocate_csm(ComponentSpace_mems *csm) {
  clReleaseMemObject(csm->ncomponents);
  clReleaseMemObject(csm->complen);
  clReleaseMemObject(csm->ele2comp);
}

void init_csm(cl_context *c, cl_program *p, cl_command_queue *q, ComponentSpace_mems *csm, cl_mem *cs, unsigned nelements) {

  int err;

  // 256 is hardcoded, but all GPUs support this currently so it isn't
  // an issue
  unsigned wgsize = 256;
  unsigned wgnum = ((nelements + wgsize - 1) / wgsize) * wgsize;

  cl_kernel kernel = clCreateKernel(*p, "init_compspace", &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating cs on device clCreateKernel => %d\n", err); exit(1);}

  err  = clSetKernelArg(kernel, 0, sizeof(void *), (void*) cs);
  err |= clSetKernelArg(kernel, 1, sizeof(void *), (void*) &(csm->ncomponents));
  err |= clSetKernelArg(kernel, 2, sizeof(void *), (void*) &(csm->complen));
  err |= clSetKernelArg(kernel, 3, sizeof(void *), (void*) &(csm->ele2comp));
  err |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*) &nelements);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating cs on device clSetKernelArg => %d\n", err); exit(1);}

  size_t global_size[3] = {wgnum, 0, 0}, local_size[3] = {wgsize, 0, 0};

  err = clEnqueueNDRangeKernel(*q,
                               kernel,
                               1,
                               NULL,
                               global_size,
                               local_size,
                               0,
                               0,
                               0);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating cs on device clCreateBuffer => %d\n", err); exit(1);}

  err = clFinish(*q);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating cs on device clCreateBuffer => %d\n", err); exit(1);}

  clReleaseKernel(kernel);
}

// Top level function to call to create and initialise a component space
cl_mem create_and_init_cs(cl_context *c, cl_program *p, cl_command_queue *q, ComponentSpace_mems *csm, unsigned nelements) {
  cl_mem ret;
  int err;
  ret = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(ComponentSpace), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating ComponentSpace on device clCreateBuffer => %d\n", err); exit(1);}

  allocate_csm(c, csm, nelements);
  init_csm(c, p, q, csm, &ret, nelements);

  return ret;
}
