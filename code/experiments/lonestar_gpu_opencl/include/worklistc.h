// The host part of the OpenCL worklist structure for GPU-Lonestar
// applications.

// OpenCL port of GPU-Lonestar applications by Tyler Sorensen (2016)

#pragma once

typedef struct {

  cl_int *dwl;
  cl_int *dnsize;
  cl_int *dindex;

} CL_Worklist2;

typedef struct {

  cl_mem dwl;
  cl_mem dnsize;
  cl_mem dindex;

} Mems_Worklist2;


cl_mem init_worklist2(cl_context *c, cl_program *p, cl_command_queue *q, int size, Mems_Worklist2* mems) {

  int err;
  int zero = 0;
  cl_kernel kernel;
  cl_mem ret;

  // Allocate device memory
  mems->dwl = clCreateBuffer(*c, CL_MEM_READ_WRITE, size * sizeof(cl_int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  mems->dnsize = clCreateBuffer(*c, CL_MEM_READ_WRITE, 1 * sizeof(cl_int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  mems->dindex = clCreateBuffer(*c, CL_MEM_READ_WRITE, 1 * sizeof(cl_int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  // Copy memory to the device
  err = clEnqueueWriteBuffer(*q, mems->dnsize, 1, 0, sizeof(cl_int), &size, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->dindex, 1, 0, sizeof(cl_int), &zero, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  // Setup the worklist memory object
  ret = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(CL_Worklist2), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  // Initialise the worklist through the 'init_worklist' kernel
  kernel = clCreateKernel(*p, "init_worklist", &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  err  = clSetKernelArg(kernel, 0, sizeof(void *), (void*) &ret);
  err |= clSetKernelArg(kernel, 1, sizeof(void *), (void*) &(mems->dwl));
  err |= clSetKernelArg(kernel, 2, sizeof(void *), (void*) &(mems->dnsize));
  err |= clSetKernelArg(kernel, 3, sizeof(void *), (void*) &(mems->dindex));
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

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
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  err = clFinish(*q);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating worklist on device clCreateBuffer => %d\n", err); exit(1);}

  clReleaseKernel(kernel);
  return ret;
}

// Reset the worklist by writing 0 to the worklist index
int wl_reset(cl_command_queue *q, Mems_Worklist2* mems) {

  int err;
  cl_int zero = 0;
  err = clEnqueueWriteBuffer(*q,
                             mems->dindex,
                             1,
                             0,
                             sizeof(cl_int),
                             &zero,
                             0,
                             0,
                             0);
  return err;
}

// Get the size of the worklist by copying back the index
int wl_get_nitems(cl_command_queue *q, Mems_Worklist2* mems, int *ret) {

  int err;
  err = clEnqueueReadBuffer(*q,
                            mems->dindex,
                            1,
                            0,
                            sizeof(cl_int),
                            ret,
                            0,
                            0,
                            0);
  return err;
}

// A nice debugging function for dumping the contents of a worklist
int wl_dump(cl_command_queue *q, Mems_Worklist2* mems) {

  int size, err;
  cl_int *data;

  err = wl_get_nitems(q, mems, &size);
  CHECK_ERR(err);

  data = (cl_int *) malloc(size * sizeof(cl_int));

  // Copy data back
  err = clEnqueueReadBuffer(*q,
                            mems->dwl,
                            1,
                            0,
                            sizeof(cl_int)*size,
                            data,
                            0,
                            0,
                            0);

  CHECK_ERR(err);

  // Print the data
  for (int i = 0; i < size; i++) {
    printf("%d\n", data[i]);
  }

  // Free the temporary data array
  free(data);
}

// Deallocate all the device buffers associated with the worklist
void dealloc_mems_wl(Mems_Worklist2* mems) {
  clReleaseMemObject(mems->dwl);
  clReleaseMemObject(mems->dnsize);
  clReleaseMemObject(mems->dindex);
}
