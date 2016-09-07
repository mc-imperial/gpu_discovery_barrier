// The host part of the OpenCL CSR graph structure for GPU-Lonestar
// applications.

// OpenCL port of GPU-Lonestar applications by Tyler Sorensen (2016)

#pragma once

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include "stdlib.h"
#include "string.h"
#include "portable_endian.h"

#include <time.h>
#include <fstream>
#include <string>
#include <iostream>
#include <limits>
#include <string.h>

#include <cassert>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <cassert>
#include <inttypes.h>

typedef struct {

  cl_uint nnodes, nedges;
  cl_uint *noutgoing, *nincoming, *srcsrc, *psrc, *edgessrcdst;
  cl_uint *edgessrcwt;
  cl_uint *maxOutDegree, *maxInDegree;

} CL_Graph;

typedef struct {

  cl_uint nnodes, nedges;
  cl_uint *noutgoing, *nincoming, *srcsrc, *psrc, *edgessrcdst;
  foru *edgessrcwt;
  cl_uint *levels;
  cl_uint source;

  cl_uint *maxOutDegree, *maxInDegree;
  cl_uint diameter;
  cl_uint foundStats;

} Graph;

typedef struct {

  cl_mem noutgoing, nincoming, srcsrc, psrc, edgessrcdst;
  cl_mem edgessrcwt;
  cl_mem maxOutDegree, maxInDegree;

} Cl_Graph_mems;

void progressPrint(Graph *g, unsigned maxii, unsigned ii) {

  const unsigned nsteps = 10;
  unsigned ineachstep = (maxii / nsteps);

  if(ineachstep == 0)
    ineachstep = 1;

  if (ii % ineachstep == 0) {
    printf("\t%3d%%\r", ii*100/maxii + 1);
    fflush(stdout);
  }
}

unsigned allocOnHost(Graph * g) {
  g->edgessrcdst = (cl_uint *) malloc((g->nedges+1) * sizeof(unsigned int)); // First entry acts as null.
  g->edgessrcwt = (foru *) malloc((g->nedges+1) * sizeof(foru));             // First entry acts as null.
  g->psrc = (cl_uint *) calloc(g->nnodes+1, sizeof(unsigned int));           // Init to null.
  g->psrc[g->nnodes] = g->nedges;                                            // Last entry points to end of edges, to avoid thread divergence in drelax.
  g->noutgoing = (cl_uint *) calloc(g->nnodes, sizeof(unsigned int));        // Init to 0.
  g->nincoming = (cl_uint *) calloc(g->nnodes, sizeof(unsigned int));        // Init to 0.
  g->srcsrc = (cl_uint *) malloc(g->nnodes * sizeof(unsigned int));

  g->maxOutDegree = (cl_uint *)malloc(sizeof(unsigned));
  g->maxInDegree = (cl_uint *)malloc(sizeof(unsigned));
  *(g->maxOutDegree) = 0;
  *(g->maxInDegree) = 0;

  return 0;
}

unsigned readFromEdges(Graph * g, char* file) {

  unsigned int prevnode = 0;
  unsigned int tempsrcnode;
  unsigned int ncurroutgoing = 0;

  std::ifstream cfile;
  cfile.open(file);

  std::string str;
  getline(cfile, str);
  sscanf(str.c_str(), "%d %d", &(g->nnodes), &(g->nedges));

  allocOnHost(g);
  for (unsigned ii = 0; ii < g->nnodes; ++ii) {
    g->srcsrc[ii] = ii;
  }

  for (unsigned ii = 0; ii < g->nedges; ++ii) {
    getline(cfile, str);
    sscanf(str.c_str(), "%d %d %d", &tempsrcnode, &(g->edgessrcdst[ii+1]), &(g->edgessrcwt[ii+1]));

    if (prevnode == tempsrcnode) {

      if (ii == 0) {
        g->psrc[tempsrcnode] = ii + 1;
      }
      ++ncurroutgoing;
    }
    else {
      g->psrc[tempsrcnode] = ii + 1;

      if (ncurroutgoing) {
        g->noutgoing[prevnode] = ncurroutgoing;
      }
      prevnode = tempsrcnode;
      ncurroutgoing = 1; // Not 0.
    }
    g->nincoming[g->edgessrcdst[ii+1]]++;
    progressPrint(g, g->nedges, ii);
  }
  g->noutgoing[prevnode] = ncurroutgoing; // Last entries.

  cfile.close();
  return 0;
}

unsigned readFromGR(Graph *g, char file[]) {

  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long masterLength = ftell(f);
  fseek(f, 0, SEEK_SET);

  void *m = (void *)malloc(masterLength + 1);
  int fread_ret = fread(m, masterLength, 1, f);

  if (fread_ret != 1) {
    printf("error in fread!!\n");
    abort();
  }
  fclose(f);

  double starttime, endtime;
  starttime = rtclock();

  // Parse file
  uint64_t* fptr = (uint64_t*)m;
  uint64_t version = le64toh(*fptr++);
  assert(version == 1);
  uint64_t sizeEdgeTy = le64toh(*fptr++);
  uint64_t numNodes = le64toh(*fptr++);
  uint64_t numEdges = le64toh(*fptr++);
  uint64_t *outIdx = fptr;
  fptr += numNodes;
  uint32_t *fptr32 = (uint32_t*)fptr;
  uint32_t *outs = fptr32;
  fptr32 += numEdges;
  if (numEdges % 2) fptr32 += 1;
  unsigned  *edgeData = (unsigned *)fptr32;

  g->nnodes = numNodes;
  g->nedges = numEdges;

  printf("nnodes=%d, nedges=%d.\n", g->nnodes, g->nedges);

  allocOnHost(g);

  for (unsigned ii = 0; ii < g->nnodes; ++ii) {
    g->srcsrc[ii] = ii;

    if (ii > 0) {
      g->psrc[ii] = le64toh(outIdx[ii - 1]) + 1;
      g->noutgoing[ii] = le64toh(outIdx[ii]) - le64toh(outIdx[ii - 1]);
    }
    else {
      g->psrc[0] = 1;
      g->noutgoing[0] = le64toh(outIdx[0]);
    }
    for (unsigned jj = 0; jj < g->noutgoing[ii]; ++jj) {
      unsigned edgeindex = g->psrc[ii] + jj;
      unsigned dst = le32toh(outs[edgeindex - 1]);

      if (dst >= g->nnodes)
        printf("\tinvalid edge from %d to %d at index %d(%d).\n", ii, dst, jj, edgeindex);

      g->edgessrcdst[edgeindex] = dst;
      g->edgessrcwt[edgeindex] = edgeData[edgeindex - 1];

      g->nincoming[dst]++;
    }
    progressPrint(g, g->nnodes, ii);
  }

  endtime = rtclock();

  printf("read %ld bytes in %0.2f ms (%0.2f MB/s)\n", masterLength, 1000 * (endtime - starttime), (masterLength / 1048576) / (endtime - starttime));

  return 0;
}

// Reads a graph from a file and stores it
// in a host Graph structure
int read_graph(Graph* g, char* file) {
  if (strstr(file, ".edges")) {
    return readFromEdges(g, file);
  } else if (strstr(file, ".gr")) {
    return readFromGR(g, file);
  }
  return 0;
}

// Create all the cl_mems needed for the device graph.
// Called from copy_graph_gpu, probably shouldn't be called
// outside of that function
void alloc_cl_mems(cl_context *c, Graph *g, Cl_Graph_mems *mems) {
  int err;

  mems->edgessrcdst = clCreateBuffer(*c, CL_MEM_READ_WRITE, (g->nedges+1) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->edgessrcwt = clCreateBuffer(*c, CL_MEM_READ_WRITE, (g->nedges+1) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->psrc = clCreateBuffer(*c, CL_MEM_READ_WRITE, (g->nnodes+1) * sizeof(cl_uint), NULL,  &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->noutgoing = clCreateBuffer(*c, CL_MEM_READ_WRITE, (g->nnodes) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->nincoming = clCreateBuffer(*c, CL_MEM_READ_WRITE, (g->nnodes) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->srcsrc = clCreateBuffer(*c, CL_MEM_READ_WRITE, (g->nnodes) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->maxOutDegree = clCreateBuffer(*c, CL_MEM_READ_WRITE, (1) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  mems->maxInDegree = clCreateBuffer(*c, CL_MEM_READ_WRITE, (1) * sizeof(cl_uint), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}
}

// Free the device buffers containing the graph
void dealloc_cl_mems(Cl_Graph_mems *mems) {
  clReleaseMemObject(mems->edgessrcdst);
  clReleaseMemObject(mems->edgessrcwt);
  clReleaseMemObject(mems->psrc);
  clReleaseMemObject(mems->noutgoing);
  clReleaseMemObject(mems->nincoming);
  clReleaseMemObject(mems->srcsrc);
  clReleaseMemObject(mems->maxOutDegree);
  clReleaseMemObject(mems->maxInDegree);
}

// Free all the host memory for the graph
void free_host_graph(Graph *g) {
  free(g->edgessrcdst);
  free(g->edgessrcwt);
  free(g->psrc);
  free(g->noutgoing);
  free(g->nincoming);
  free(g->srcsrc);
  free(g->maxOutDegree);
  free(g->maxInDegree);
}

// This function creates and initialises a graph on the device
cl_mem create_and_init_gpu_graph(cl_context *c, cl_program *prog, cl_command_queue *q, Graph *g, Cl_Graph_mems *mems) {

  int err;
  cl_kernel kernel;
  cl_mem ret = clCreateBuffer(*c, CL_MEM_READ_WRITE, sizeof(CL_Graph), NULL, &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  // Initialise the graph with a kernel "init_graph"
  kernel = clCreateKernel(*prog, "init_graph", &err);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  err  = clSetKernelArg(kernel, 0, sizeof(void *), (void*) &ret);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_uint), (void *) &(g->nnodes));
  err |= clSetKernelArg(kernel, 2, sizeof(cl_uint), (void *) &(g->nedges));
  err |= clSetKernelArg(kernel, 3, sizeof(void *), (void *) &(mems->noutgoing));
  err |= clSetKernelArg(kernel, 4, sizeof(void *), (void *) &(mems->nincoming));
  err |= clSetKernelArg(kernel, 5, sizeof(void *), (void *) &(mems->srcsrc));
  err |= clSetKernelArg(kernel, 6, sizeof(void *), (void *) &(mems->psrc));
  err |= clSetKernelArg(kernel, 7, sizeof(void *), (void *) &(mems->edgessrcdst));
  err |= clSetKernelArg(kernel, 8, sizeof(void *), (void *) &(mems->edgessrcwt));
  err |= clSetKernelArg(kernel, 9, sizeof(void *), (void *) &(mems->maxOutDegree));
  err |= clSetKernelArg(kernel, 10, sizeof(void *), (void *) &(mems->maxInDegree));
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

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
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  err = clFinish(*q);
  if(err != CL_SUCCESS) { fprintf(stderr, "ERROR: allocating graph on device clCreateBuffer => %d\n", err); exit(1);}

  clReleaseKernel(kernel);
  return ret;
}

// This function copies the host graph to a cl_mem graph on the device
cl_mem copy_graph_gpu(cl_context *c, cl_program *p, cl_command_queue *q, Graph *g, Cl_Graph_mems *mems) {

  int err;
  alloc_cl_mems(c, g, mems);

  err = clEnqueueWriteBuffer(*q, mems->edgessrcdst, 1, 0, (g->nedges+1) * sizeof(cl_uint), g->edgessrcdst, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->edgessrcwt, 1, 0, (g->nedges+1) * sizeof(cl_uint), g->edgessrcwt, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->psrc, 1, 0, (g->nnodes+1) * sizeof(cl_uint), g->psrc, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->noutgoing, 1, 0, (g->nnodes) * sizeof(cl_uint), g->noutgoing, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->nincoming, 1, 0, (g->nnodes) * sizeof(cl_uint), g->nincoming, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->srcsrc, 1, 0, (g->nnodes) * sizeof(cl_uint), g->srcsrc, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->maxOutDegree, 1, 0, (1) * sizeof(cl_uint), g->maxOutDegree, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  err = clEnqueueWriteBuffer(*q, mems->maxInDegree, 1, 0, (1) * sizeof(cl_uint), g->maxInDegree, 0, 0, 0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: copying graph to gpu: %d\n", err); exit(1);}

  return create_and_init_gpu_graph(c, p, q, g, mems);
}
