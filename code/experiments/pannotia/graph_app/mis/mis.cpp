/************************************************************************************\
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

// File modified by Tyler Sorensen (2016) to run the experiments for
// the OOPSLA'16 paper "Portable Inter-Workgroup Barrier
// Synchronisation for GPUs"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include <algorithm>
#include "parse.h"
#include "util.h"
#include "my_opencl.h"

#define RANGE 2048

int initialize(int use_gpu);
int shutdown();
void dump2file(int *adjmatrix, int num_nodes);
void print_vector(int *vector, int num);
void print_vectorf(float *vector, int num);

// Local OpenCL utilities
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id     target_device;
static cl_device_id   * device_list;
static cl_int           num_devices;

const char * CL_FILE = STRINGIFY(KERNEL_DIR) "mis_kernel.cl";

#define EXE_NAME "mis"

int main(int argc, char **argv) {

  char *tmpchar;
  int num_nodes;
  int num_edges;
  int use_gpu = 1;
  int file_format = 1;
  bool directed = 0;
  cl_int err = 0;
  int wgs;

  if (argc == 4) {
    tmpchar = argv[1];           // Graph file
    file_format = atoi(argv[2]); // Graph format
    wgs = atoi(argv[3]);         // Threads per workgroup
  }
  else {
    printf("invalid usage:\n");
    printf("./" EXE_NAME " GRAPH_FILE GRAPH_FORMAT WORKGROUP_SIZE\n\n");
    printf("linux example:\n");
    printf("./" EXE_NAME " ../dataset/color/G3_circuit.graph  1 256\n");
    printf("windows VS example:\n");
    printf(EXE_NAME " ..\\..\\dataset\\color\\G3_circuit.graph  1 256\n");
    exit(1);
  }

  // Allocate the CSR array
  csr_array *csr = (csr_array *)malloc(sizeof(csr_array));
  if (!csr) fprintf(stderr, "malloc failed csr\n");

  // Parse the graph into the CSR structure
  if (file_format == 1)
    csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
  else if (file_format == 0)
    csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);
  else {
    fprintf(stderr, "reserve for future");
    exit(1);
  }

  // Allocate the node value array
  float *node_value = (float *) malloc(num_nodes * sizeof(float));
  if (!node_value) fprintf(stderr, "malloc failed node_value\n");

  // Allocate the set array
  int *s_array = (int *) malloc(num_nodes * sizeof(int));
  if (!s_array) fprintf(stderr, "malloc failed s_array\n");

  // Initialise nodes.
  // Original application used random values, we use
  // determinisic values for reproducable runtimes.
  for (int i = 0; i < num_nodes; i++) {

    // Original application: node_value[i] =  rand()/(float)RAND_MAX;
    node_value[i] = float(i/((float)(num_nodes + 1)));
  }

  // Load the OpenCL kernel file
  int sourcesize = 1024 * 1024;
  char * source = (char *) calloc(sourcesize, sizeof(char));
  if (!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

  FILE * fp = fopen(CL_FILE, "rb");
  if (!fp) { printf("ERROR: unable to open '%s'\n", CL_FILE); return -1; }
  fread(source + strlen(source), sourcesize, 1, fp);
  fclose(fp);

  // Initialize the OpenCL variables
  if (initialize(use_gpu)) return -1;

  char opts[500];
  get_compile_opts(opts);
  cl_program prog = build_program(context, target_device, CL_FILE, opts);

  // Create OpenCL kernels
  cl_kernel kernel1, kernel2, kernel3, kernel4;
  const char * kernelpr1  = "init";
  const char * kernelpr2  = "mis1";
  const char * kernelpr3  = "mis2";
  const char * kernelpr4  = "mis3";

  kernel1 = clCreateKernel(prog, kernelpr1, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 1 => %d\n", err); return -1; }

  kernel2 = clCreateKernel(prog, kernelpr2, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 2 => %d\n", err); return -1; }

  kernel3 = clCreateKernel(prog, kernelpr3, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 3 => %d\n", err); return -1; }

  kernel4 = clCreateKernel(prog, kernelpr4, &err);
  if (err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 4 => %d\n", err); return -1; }

  clReleaseProgram(prog);

  // Device buffers
  cl_mem row_d, col_d, c_array_d, c_array_u_d,
    s_array_d, node_value_d, min_array_d, stop_d;

  // Allocate the device-side buffers for the graph
  row_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer row_d (size:%d) => %d\n", num_nodes, err); return -1;}

  col_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_edges * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer col_d (size:%d) => %d\n", num_edges, err); return -1;}

  // Termination variable
  stop_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer stop_d (size:%d) => %d\n", 1, err); return -1;}

  // mis buffers
  min_array_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer min_array_d (size:%d) => %d\n", num_nodes, err); return -1;}

  c_array_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer c_array_d (size:%d) => %d\n", num_nodes, err); return -1;}

  c_array_u_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer c_array_d (size:%d) => %d\n", num_nodes, err); return -1;}

  s_array_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer s_array_d (size:%d) => %d\n", num_nodes, err); return -1;}

  node_value_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer node_value_d (size:%d) => %d\n", num_nodes, err); return -1;}

  // Copy data to device-side buffers
  err = clEnqueueWriteBuffer(cmd_queue,
                             row_d,
                             1,
                             0,
                             num_nodes * sizeof(int),
                             csr->row_array,
                             0,
                             0,
                             0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer row_d (size:%d) => %d\n", num_nodes, err); return -1; }

  err = clEnqueueWriteBuffer(cmd_queue,
                             col_d,
                             1,
                             0,
                             num_edges * sizeof(int),
                             csr->col_array,
                             0,
                             0,
                             0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer col_d (size:%d) => %d\n", num_nodes, err); return -1; }

  err = clEnqueueWriteBuffer(cmd_queue,
                             node_value_d,
                             1,
                             0,
                             num_nodes * sizeof(float),
                             node_value,
                             0,
                             0,
                             0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer feature_d (size:%d) => %d\n", num_nodes, err); return -1; }

  // Kernel dimensions
  int block_size = wgs;
  int global_size = (num_nodes % block_size == 0) ? num_nodes: (num_nodes/block_size + 1) * block_size;

  size_t local_work[3]  = { block_size,  1, 1 };
  size_t global_work[3] = { global_size, 1, 1 };

  // --Initialise

  // Set kernel args for init kernel
  clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &s_array_d);
  clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &c_array_d);
  clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &c_array_u_d);
  clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &num_nodes);
  clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*) &num_edges);

  // Launch the initalization kernel
  err = clEnqueueNDRangeKernel(cmd_queue,
                               kernel1,
                               1,
                               NULL,
                               global_work,
                               local_work,
                               0,
                               0,
                               0);

  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel1 (%d)\n", err); return -1; }

  // --Set up kernel args

  // mis1
  clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &row_d);
  clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &col_d);
  clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &node_value_d);
  clSetKernelArg(kernel2, 3, sizeof(void *), (void*) &s_array_d);
  clSetKernelArg(kernel2, 4, sizeof(void *), (void*) &c_array_d);
  clSetKernelArg(kernel2, 5, sizeof(void *), (void*) &min_array_d);
  clSetKernelArg(kernel2, 6, sizeof(void *), (void*) &stop_d);
  clSetKernelArg(kernel2, 7, sizeof(cl_int), (void*) &num_nodes);
  clSetKernelArg(kernel2, 8, sizeof(cl_int), (void*) &num_edges);

  // mis2
  clSetKernelArg(kernel3, 0, sizeof(void *), (void*) &row_d);
  clSetKernelArg(kernel3, 1, sizeof(void *), (void*) &col_d);
  clSetKernelArg(kernel3, 2, sizeof(void *), (void*) &node_value_d);
  clSetKernelArg(kernel3, 3, sizeof(void *), (void*) &s_array_d);
  clSetKernelArg(kernel3, 4, sizeof(void *), (void*) &c_array_d);
  clSetKernelArg(kernel3, 5, sizeof(void *), (void*) &c_array_u_d);
  clSetKernelArg(kernel3, 6, sizeof(void *), (void*) &min_array_d);
  clSetKernelArg(kernel3, 7, sizeof(cl_int), (void*) &num_nodes);
  clSetKernelArg(kernel3, 8, sizeof(cl_int), (void*) &num_edges);

  // mis3
  clSetKernelArg(kernel4, 0, sizeof(void *), (void*) &c_array_u_d);
  clSetKernelArg(kernel4, 1, sizeof(void *), (void*) &c_array_d);
  clSetKernelArg(kernel4, 2, sizeof(cl_int), (void*) &num_nodes);

  // Termination variable
  int stop = 1;

  double time1 = gettime();
  while (stop) {

    stop = 0;

    // Copy the termination variable to the device
    err = clEnqueueWriteBuffer(cmd_queue, stop_d, 1, 0, sizeof(int), &stop, 0, 0, 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: write stop_d variable (%d)\n", err); return -1; }

    // Launch mis1
    err = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel2,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel2 (%d)\n", err); return -1; }

    // Launch mis2
    err = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel3,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel3 (%d)\n", err); return -1; }

    // Launch mis3
    err = clEnqueueNDRangeKernel(cmd_queue,
                                 kernel4,
                                 1,
                                 NULL,
                                 global_work,
                                 local_work,
                                 0,
                                 0,
                                 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: kernel4 (%d)\n", err); return -1; }

    clFinish(cmd_queue);

    // Copy the termination variable back
    err = clEnqueueReadBuffer(cmd_queue, stop_d, 1, 0, sizeof(int), &stop, 0, 0, 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read stop_d variable (%d)\n", err); return -1; }
  }

  clFinish(cmd_queue);
  double time2 = gettime();

  err = clEnqueueReadBuffer(cmd_queue,
                            s_array_d,
                            1,
                            0,
                            num_nodes * sizeof(int),
                            s_array,
                            0,
                            0,
                            0);

  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueReadBuffer()=>%d failed\n", err); return -1; }

  // Print out the timing info
  printf("kernel time = %f ms\n", (time2 - time1) * 1000);

  // Print the set array to file
  print_vector(s_array, num_nodes);

  // Clean up the host-side arrays
  free(node_value);
  free(s_array);
  csr->freeArrays();
  free(csr);

  // Clean up the device-side arrays
  clReleaseMemObject(row_d);
  clReleaseMemObject(col_d);
  clReleaseMemObject(c_array_d);
  clReleaseMemObject(s_array_d);
  clReleaseMemObject(node_value_d);
  clReleaseMemObject(min_array_d);
  clReleaseMemObject(stop_d);

  // Clean up the OpenCL variables
  shutdown();
  print_device_info();
  return 0;
}

void print_vector(int *vector, int num) {

  FILE * fp = fopen("mis.out", "w");
  if (!fp) { printf("ERROR: unable to open result.txt\n");}

  for (int i = 0; i < num; i++) {
    fprintf(fp, "%d\n", vector[i]);
  }
  fclose(fp);
}

void print_vectorf(float *vector, int num) {

  FILE * fp = fopen("result.out", "w");
  if (!fp) { printf("ERROR: unable to open result.txt\n");}

  for (int i = 0; i < num; i++) {
    fprintf(fp, "%f\n", vector[i]);
  }

  fclose(fp);
}

int initialize(int use_gpu) {
  cl_int result;
  size_t size;

  // Create OpenCL context
  cl_platform_id platform_id;
  if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
  cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
  device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

  context = clCreateContextFromType( ctxprop,
                                     device_type,
                                     NULL,
                                     NULL,
                                     NULL );

  if (!context) { fprintf(stderr, "ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU"); return -1; }

  // Get the list of GPUs
  result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size);
  num_devices = (int) (size / sizeof(cl_device_id));
  printf("num_devices = %d\n", num_devices);

  if (result != CL_SUCCESS || num_devices < 1) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }
  device_list = new cl_device_id[num_devices];
  if (!device_list) { fprintf(stderr, "ERROR: new cl_device_id[] failed\n"); return -1; }
  result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL);
  if (result != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }

  target_device = device_list[REQUESTED_DEVICE];

  // Create command queue for the first device
  cmd_queue = clCreateCommandQueue( context,
                                    target_device,
                                    0,
                                    NULL );

  if (!cmd_queue) { fprintf(stderr, "ERROR: clCreateCommandQueue() failed\n"); return -1; }
  return 0;
}

int shutdown() {
  // Release resources
  if (cmd_queue) clReleaseCommandQueue(cmd_queue);
  if (context) clReleaseContext(context);
  if (device_list) delete device_list;

  // Reset all variables
  cmd_queue = 0;
  context = 0;
  device_list = 0;
  num_devices = 0;
  device_type = 0;
  return 0;
}
