// A utility file to build OpenCL programs, create devices, print device info
// and get compiler options that are useful for the OOPSLA benchmarks.

#pragma once

#include <CL/cl.h>
#include "stdio.h"
#include <stdlib.h>
#include "string.h"

#define STRINGIFY_INNER(X) #X
#define STRINGIFY(X) STRINGIFY_INNER(X)

typedef cl_int my_int;

// Define these using -D if you have more than 2 devices
// on your platform
#ifndef MAX_DEVICES
#define MAX_DEVICES 2
#endif

// Define these using -D if you want to use a device other
// than device 0
#ifndef REQUESTED_DEVICE
#define REQUESTED_DEVICE 0
#endif

#define EXIT_FAILURE 1
#define SAFE_CALL(call) do {			\
    int SAFE_CALL_ERR = call;			\
    if(SAFE_CALL_ERR < 0) {			\
      printf("error in file '%s' in line %i\n",	\
	     __FILE__, __LINE__);		\
      exit(EXIT_FAILURE);			\
    } } while (0)

// Creates a device from platform 0 and using REQUESTED_DEVICE.
// Probably should be made more modular to select particular
// platforms and devices.
cl_device_id create_device() {

  cl_platform_id platform;
  cl_device_id dev[MAX_DEVICES];
  cl_uint num_devices;

  SAFE_CALL(clGetPlatformIDs(1, &platform, NULL));
  SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, MAX_DEVICES, dev, &num_devices));

  if(REQUESTED_DEVICE >= num_devices) {
    perror("Requested device not available.");
    exit(1);
  }

  return dev[REQUESTED_DEVICE];
}

// Given a file name, options, and an OpenCL context and device: build the program.
// Outputs compilation errors if they are encountered
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename, const char* options) {

  cl_program program;
  FILE *program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;
  int err = 0;

  program_handle = fopen(filename, "r");
  if(program_handle == NULL) {
    perror("Couldn't find the program file");
    exit(1);
  }
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  program = clCreateProgramWithSource(ctx, 1,
				      (const char** )&program_buffer, &program_size, &err);
  if(err < 0) {
    perror("Couldn't create the program");
    exit(1);
  }
   

  err = clBuildProgram(program, 1, &dev, options, NULL, NULL);
  if(err < 0) {

    // Find size of log and print to std output
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			  0, NULL, &log_size);
    program_log = (char* ) malloc(log_size + 1);

    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
			  log_size + 1, program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(1);
  }

  free(program_buffer);
  return program;
}

// Utility function to parse a char * into an integer
int parse_int(char *p) {
  int intvar;
  if (sscanf (p, "%i", &intvar)!=1) { 
    printf("error - %s not an integer\n", p); 
    exit(1);
  }
  return intvar;
}

// Get compile options. If OpenCL 2.0 is available, then the
// built-in atomics are used. Otherwise use custom atomics.
// Additionally some compiler bugs (??) require a different
// loop structure to be used. We define them here.
void get_compile_opts(char * opts) {
  cl_device_id device = create_device();
  char buffer[512];

  opts[0] = '\0';
  
  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
  if (strstr(buffer, "OpenCL 2.0") != 0) {
    strcat(opts, "-cl-std=CL2.0");
  }
  else {
    strcat(opts, "-DCUSTOM_ATOMICS");
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
    if (strcmp("NVIDIA Corporation", buffer) == 0) {
      strcat(opts, " -DNVIDIA");
    }    
    if (strcmp("ARM", buffer) == 0) {
      strcat(opts, " -DARM");
    }
    // This is to experiment with intel chips that are not OpenCL 2.0.
    // The GPUs don't seem to work. The CPUs do seem to work.
    if (strcmp("Intel(R) Corporation", buffer) == 0) {
      strcat(opts, " -DARM");
    }
  }
  strcat(opts, " -I");
  strcat(opts, STRINGIFY(CL_ACTIVE_GROUP_PATH));

  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
  if (strcmp("Quadro K5200", buffer) == 0) {
    strcat(opts, " -DNO_MIS_LOOP");
  }
  if (strcmp("Tonga", buffer) == 0) {
    strcat(opts, " -DNO_MIS_LOOP");
  }
  if (strcmp("Spectre", buffer) == 0) {
    strcat(opts, " -DNO_MIS_LOOP");
  }

  strcat(opts, " -DINT_TYPE=int");
  strcat(opts, " -DATOMIC_INT_TYPE=atomic_int");

#if defined(LONESTAR_CL_INCLUDE)
  strcat(opts, " -I");
  strcat(opts, STRINGIFY(LONESTAR_CL_INCLUDE));

  strcat(opts, " -I");
  strcat(opts, STRINGIFY(KERNEL_DIR));
#endif

}

// Get compiler options for when the workgroup
// size has to be defined with a -D option
void get_compile_opts_wgs(char * opts, int wgs) {
  char str[30];
  get_compile_opts(opts);
  sprintf(str, " -DWGS=%d", wgs);
  strcat(opts, str);
}

// Get the compile options for testing different mutex
// implementations for occupancy_tests
void get_compile_opts_occupancy_tests(char * opts, int bak) {
  get_compile_opts(opts);
  if (bak == 0) {
    strcat(opts, " -DSPIN_LOCK");
  }
}

// Print useful information about the device. 
void print_device_info() {
  cl_device_id device = create_device();
  char buffer[512];
  cl_uint buf_uint;
  cl_ulong buf_ulong;
  printf("\n  -- device info --\n");
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
  printf("DEVICE_NAME:                %s\n", buffer);
  clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
  printf("DEVICE_VENDOR:              %s\n", buffer);
  clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
  printf("DEVICE_VERSION:             %s\n", buffer);
  clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
  printf("DRIVER_VERSION:             %s\n", buffer);
  clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL);
  printf("DEVICE_MAX_COMPUTE_UNITS:   %u\n", (unsigned int)buf_uint);
  clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL);
  printf("DEVICE_MAX_CLOCK_FREQUENCY: %u\n", (unsigned int)buf_uint);
  clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
  printf("DEVICE_GLOBAL_MEM_SIZE:     %llu\n", (unsigned long long)buf_ulong);
  clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
  printf("DEVICE_LOCAL_MEM_SIZE:      %llu\n", (unsigned long long)buf_ulong);
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(buf_ulong), &buf_ulong, NULL);
  printf("DEVICE_MAX_WORK_GROUP_SIZE: %llu\n", (unsigned long long)buf_ulong);
}
