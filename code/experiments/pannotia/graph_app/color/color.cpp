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
#include "parse.h"
#include "util.h"
#include "my_opencl.h"

int initialize(int use_gpu);
int shutdown();
void print_vector(int *vector, int num, const char* fname);

const char * CL_FILE = STRINGIFY(KERNEL_DIR) "color_kernel.cl";

#define EXE_NAME "color"

// Local OpenCL utilities
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_device_id     target_device;
static cl_int           num_devices;

int main(int argc, char **argv) {

  char *tmpchar;
  char *filechar;

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

  // Deterministic seed for benchmarking
  srand(7);

  // Allocate the CSR structure
  csr_array *csr = (csr_array *) malloc(sizeof(csr_array));
  if (!csr) fprintf(stderr, "csr array malloc failed\n");

  // Parse graph file and store into a CSR format
  if (file_format == 1)
    csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
  else if (file_format == 0)
    csr = parseCOO(tmpchar, &num_nodes, &num_edges, directed);
  else {
    printf("reserve for future");
    exit(1);
  }

  // Alloate the vertex value array
  float *node_value = (float *) malloc(num_nodes * sizeof(float));
  if(!node_value) fprintf(stderr, "node_value malloc failed\n");

  // Allocate the color array
  int *color = (int *) malloc(num_nodes * sizeof(int));
  if (!color) fprintf(stderr, "color malloc failed\n");

  // Initialize all the colors to -1.
  // Original application used random values, we use
  // determinisic values for reproducable runtimes.
  for (int i = 0; i < num_nodes; i++) {
    color[i] =  -1;

    // Original application: Node_value[i] =  rand()/(float)RAND_MAX;
    node_value[i] =  i/(float)(num_nodes + 1);
  }

  // Load the OpenCL kernel file
  int sourcesize = 1024 * 1024;
  char * source = (char *) calloc(sourcesize, sizeof(char));
  if (!source) { fprintf(stderr, "ERROR: calloc(%d) failed\n", sourcesize); return -1; }

  FILE * fp = fopen(CL_FILE, "rb");
  if (!fp) { printf("ERROR: unable to open '%s'\n", CL_FILE); return -1; }
  fread(source + strlen(source), sourcesize, 1, fp);
  fclose(fp);

  // OpenCL initialization
  if(initialize(use_gpu)) return -1;

  // Create the OpenCL program
  const char * slist[2] = { source, 0 };
  char opts[500];
  get_compile_opts(opts);
  cl_program prog = build_program(context, target_device, CL_FILE, opts);

  // Create kernel files
  cl_kernel kernel1, kernel2;
  const char * kernelpr1  = "color";
  const char * kernelpr2  = "color2";

  kernel1 = clCreateKernel(prog, kernelpr1, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 1 => %d\n", err); return -1; }
  kernel2 = clCreateKernel(prog, kernelpr2, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateKernel() 2 => %d\n", err); return -1; }

  clReleaseProgram(prog);

  // Device buffers
  cl_mem row_d, col_d, max_d, color_d, node_value_d, stop_d;

  // Device-side buffers for the graph
  row_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer row_d (size:%d) => %d\n", num_nodes, err); return -1;}

  col_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_edges * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer col_d (size:%d) => %d\n", num_edges, err); return -1;}

  // Termination variables
  stop_d = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer stop_d (size:%d) => %d\n", 1, err); return -1;}

  // Buffers for color
  color_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(int), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer color_d (size:%d) => %d\n", num_nodes, err); return -1;}

  node_value_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer node_value_d (size:%d) => %d\n", num_nodes, err); return -1;}

  max_d = clCreateBuffer(context, CL_MEM_READ_WRITE, num_nodes * sizeof(float), NULL, &err);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clCreateBuffer max_d (size:%d) => %d\n", num_nodes, err); return -1;}

  // Copy data to device-side buffers
  double timer1 = gettime();

  err = clEnqueueWriteBuffer(cmd_queue,
                             color_d,
                             1,
                             0,
                             num_nodes * sizeof(int),
                             color,
                             0,
                             0,
                             0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer color_d (size:%d) => %d\n", num_nodes, err); return -1; }

  err = clEnqueueWriteBuffer(cmd_queue,
                             max_d,
                             1,
                             0,
                             num_nodes * sizeof(int),
                             color,
                             0,
                             0,
                             0);
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer max_d (size:%d) => %d\n", num_nodes, err); return -1; }

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
  if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: clEnqueueWriteBuffer node_value_d (size:%d) => %d\n", num_nodes, err); return -1; }

  //set up kernel dimensions
  int block_size = wgs;
  int global_size = (num_nodes % block_size == 0) ? num_nodes: (num_nodes/block_size + 1) * block_size;
  size_t local_work[3]  = { block_size,  1,  1 };
  size_t global_work[3] = { global_size, 1,  1 };

  int stop = 1;
  int graph_color = 1;

  // --Set up kernel args

  // color
  clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &row_d);
  clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &col_d);
  clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &node_value_d);
  clSetKernelArg(kernel1, 3, sizeof(void *), (void*) &color_d);
  clSetKernelArg(kernel1, 4, sizeof(void *), (void*) &stop_d);
  clSetKernelArg(kernel1, 5, sizeof(void *), (void*) &max_d);
  clSetKernelArg(kernel1, 7, sizeof(cl_int), (void*) &num_nodes);
  clSetKernelArg(kernel1, 8, sizeof(cl_int), (void*) &num_edges);

  // color2
  clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &node_value_d);
  clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &color_d);
  clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &max_d);
  clSetKernelArg(kernel2, 4, sizeof(cl_int), (void*) &num_nodes);
  clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*) &num_edges);

  // Main computation loop
  double timer3 = gettime();

  while (stop) {
    stop = 0;

    // Copy the termination variable to the device
    err = clEnqueueWriteBuffer(cmd_queue, stop_d, 1, 0, sizeof(int), &stop, 0, 0, 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: write stop_d (%d)\n", err); }

    clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &graph_color);
    clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &graph_color);

    // Launch 'color'
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

    // Launch 'color2'
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

    err = clEnqueueReadBuffer(cmd_queue, stop_d, 1, 0, sizeof(int), &stop, 0, 0, 0);
    if (err != CL_SUCCESS) { fprintf(stderr, "ERROR: read stop_d (%d)\n", err);}

    // Increment for the next iteration
    graph_color++;

  }
  clFinish(cmd_queue);

  double timer4 = gettime();

  // Copy back the color array
  err = clEnqueueReadBuffer(cmd_queue,
                            color_d,
                            1,
                            0,
                            num_nodes * sizeof(int),
                            color,
                            0,
                            0,
                            0);

  if(err != CL_SUCCESS) { printf("ERROR: clEnqueueReadBuffer()=>%d failed\n", err); return -1; }

  double timer2 = gettime();

  // Print color and timing info
  printf("total number of colors used: %d\n", graph_color);
  printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
  printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

  // Dump the color array into an output file
  print_vector(color, num_nodes, "color.out");

  // Free host-side buffers
  free(node_value);
  free(color);
  csr->freeArrays();
  free(csr);

  // Free OpenCL buffers
  clReleaseMemObject(row_d);
  clReleaseMemObject(col_d);
  clReleaseMemObject(max_d);
  clReleaseMemObject(color_d);
  clReleaseMemObject(node_value_d);
  clReleaseMemObject(stop_d);

  // Cleanup OpenCL variables
  shutdown();
  print_device_info();
  return 0;
}

void print_vector(int *vector, int num, const char * fname) {

  FILE * fp = fopen(fname, "w");
  if(!fp) { printf("ERROR: unable to open result.txt\n");}

  for (int i = 0; i < num; i++)
    fprintf(fp, "%d: %d\n", i + 1, vector[i]);

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
  result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
  num_devices = (int) (size / sizeof(cl_device_id));
  printf("num_devices = %d\n", num_devices);
  if (result != CL_SUCCESS || num_devices < 1) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }

  device_list = new cl_device_id[num_devices];
  if (!device_list) { fprintf(stderr, "ERROR: new cl_device_id[] failed\n"); return -1; }

  result = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
  if(result != CL_SUCCESS) { fprintf(stderr, "ERROR: clGetContextInfo() failed\n"); return -1; }

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
