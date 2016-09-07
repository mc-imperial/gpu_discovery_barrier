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

// Copied by hand in from REF-RESULTS file to check the result
void check_outcome(char* input, unsigned weight, unsigned components, unsigned edges) {
  printf("validating...\n");
  if (strstr(input,"USA-road-d.FLA.sym.gr") != NULL) {
    assert(weight == 1806814846);
    assert(components == 1);
    assert(edges == 1070375);
    printf("validation passed\n");
    return;
  }
  if (strstr(input,"rmat12.sym.gr") != NULL) {
    assert(weight == 2560798);
    assert(components == 154);
    assert(edges == 3942);
    printf("validation passed\n");
    return;
  }
  if (strstr(input,"2d-2e20.sym.gr") != NULL) {
    assert(weight == 2806989831);
    assert(components == 1);
    assert(edges == 1048575);
    printf("validation passed\n");
    return;
  }
  printf("unable to validate\n");
  assert(0);
}

// Allocate and initialise the device buffers
void allocate_cl_mems(cl_context *c, cl_command_queue *q, Graph g) {
  int err;
  mstwt = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  eleminwts = clCreateBuffer(*c, CL_MEM_READ_WRITE, g.nnodes * sizeof(foru), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  minwtcomponent = clCreateBuffer(*c, CL_MEM_READ_WRITE, g.nnodes * sizeof(foru), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  phores = clCreateBuffer(*c, CL_MEM_READ_WRITE, g.nnodes * sizeof(cl_uint), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  partners = clCreateBuffer(*c, CL_MEM_READ_WRITE, g.nnodes * sizeof(cl_uint), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  goaheadnodeofcomponent = clCreateBuffer(*c, CL_MEM_READ_WRITE, g.nnodes * sizeof(cl_uint), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  processinnextiteration = clCreateBuffer(*c, CL_MEM_READ_WRITE, g.nnodes * sizeof(cl_int), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  grepeat = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_int), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  // Repeat starts out as false
  int repeat = 0;
  err = clEnqueueWriteBuffer(*q,
                             grepeat,
                             1,
                             0,
                             sizeof(cl_int),
                             &repeat,
                             0,
                             0,
                             0);

  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  gedgecount = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(cl_uint), NULL, &err);
  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  // Edgecount starts out as 0
  unsigned edgecount = 0;
  err = clEnqueueWriteBuffer(*q,
                             gedgecount,
                             1,
                             0,
                             sizeof(cl_uint),
                             &edgecount,
                             0,
                             0,
                             0);

  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }

  // weight starts out as 0
  err = clEnqueueWriteBuffer(*q,
                             mstwt,
                             1,
                             0,
                             sizeof(cl_uint),
                             &edgecount,
                             0,
                             0,
                             0);

  if (err < 0 ) { printf("failed allocating gpu %d\n", err); exit(1); }
}

// Release the device side buffers
void release_cl_mems() {
  clReleaseMemObject(mstwt);
  clReleaseMemObject(eleminwts);
  clReleaseMemObject(minwtcomponent);
  clReleaseMemObject(partners);
  clReleaseMemObject(phores);
  clReleaseMemObject(processinnextiteration);
  clReleaseMemObject(grepeat);
  clReleaseMemObject(gedgecount);
  clReleaseMemObject(goaheadnodeofcomponent);
}
