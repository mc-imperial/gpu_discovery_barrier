// Program to test the occupancy of the GPU.
// Takes in the number of workgroups, size of workgroups, amount of local memory, 
// a flag if the protocol is enabled, and a flag for which mutex to use. 
// Reports the number of discovered groups (or potentially deadlocks if the 
// protocol is disabled and the requested resources is too much to run concurrently). 

#include "stdio.h"
#include "stdlib.h"

#include "my_opencl.h"
#include "discovery.h"

cl_device_id device;
cl_context context;
cl_program program;
cl_command_queue queue;
cl_kernel kernel;

char * CL_FILE= STRINGIFY(KERNEL_DIR) "occupancy_test.cl";

int main(int argc, char **argv) {

  int wgc, wgs, lms, prot, ticket;
  int err;

  if (argc != 6) {
    printf("please provide number of workgroups, workgroup size, local memory size, flag for protocol, flag for ticket lock\n");
    return 0;
  }
  
  wgc = parse_int(argv[1]);
  wgs = parse_int(argv[2]);
  lms = parse_int(argv[3]);
  prot = parse_int(argv[4]);
  ticket = parse_int(argv[5]);
  printf("running with\nworkgroup count: %d\nworkgroup size: %d\nlocal memory size: %d\nprotocol enabled: %d\nusing ticket lock: %d\n", wgc, wgs,lms,prot,ticket);

  device = create_device();
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  if (err < 0 ) { perror("Couldn't create OpenCL context"); exit(1); }

  char opts[200];
  get_compile_opts_occupancy_tests(opts, ticket);
  printf("compiler options are: %s\n", opts);

  program = build_program(context, device, CL_FILE, opts);

  kernel = clCreateKernel(program, "run_test",&err);
  if (err < 0 ) { perror("Couldn't get kernel run_test_static"); exit(1); }

  cl_mem d_gl_ctx;
  d_gl_ctx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(discovery_kernel_ctx), NULL ,&err);
  if (err < 0 ) { perror("Couldn't create buffer"); exit(1); }
      
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
  if (err < 0 ) { perror("failed create command queue barrier"); exit(1); }

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_gl_ctx);
  err |= clSetKernelArg(kernel, 1, lms, NULL);
  if (err < 0 ) { printf("error set_arg0 %d\n", err); exit(1); }

  err = init_discovery_kernel_ctx_skip(&program, &queue, &d_gl_ctx, !prot);
  if (err < 0) { perror("failed kernel context"); exit(1); }

  size_t global_size = wgs * wgc, local_size = wgs;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
  if (err < 0 ) { return err; }

  int participating_groups = number_of_participating_groups(&queue, &d_gl_ctx);
  printf("kernel ran with a total of %d workgroups\n", participating_groups);

  // Cleanup
  err = clReleaseMemObject(d_gl_ctx);
  if (err != CL_SUCCESS) {perror("OpenCL Error"); exit(1);}

  err = clReleaseCommandQueue(queue);
  if (err != CL_SUCCESS) {perror("OpenCL Error"); exit(1);}

  err = clReleaseProgram(program);
  if (err != CL_SUCCESS) {perror("OpenCL Error"); exit(1);}

  err = clReleaseContext(context);
  if (err != CL_SUCCESS) {perror("OpenCL Error"); exit(1);}
  
  printf("\n\n");
  print_device_info();
  
  return 0;
}
