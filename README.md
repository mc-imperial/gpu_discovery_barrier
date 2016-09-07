Author: Tyler Sorensen 

Last revised: 1 Sept. 2016

# Table of Contents

- Introduction

- Licenses and Copyright

- Running the Code
   - Obtaining Inputs
   - Running

- Known Issues

# Introduction

This is the code for the paper:

"Portable Inter-Workgroup Barrier Synchronisation for GPUs" appearing
in OOPSLA'16.

Please cite the above paper if you use this code in academic work.

If you use our Pannotia or LonestarGPU code, consider citing the papers
for those works as well, which are respectively:

"Pannotia: Understanding irregular GPGPU graph applications"
appearing in IISWC'13

"A quantitative study of irregular programs on GPUs" appearing in
IISW'12

Please contact Ally or Tyler if have questions or comments about this
code:

Tyler Sorensen 
t.sorensen15 AT imperial DOT ac DOT uk

Alastair F. Donaldson
alastair.donaldson AT imperial DOT ac DOT uk

# Licenses and Copyright

This code consists of 3 different code bases, all with different
licenses and copyright holders.

### Discovery Protocol

The original code written for this project consists of the discovery
protocol, the XF barrier (written using the discovery protocol and
OpenCL 2.0 atomics) and the occupancy micro-benchmarks.

This code is located in 
code/discovery_protocol
/code/experiments/occupancy_tests

The copyright holders of this code are: Tyler Sorensen and Alastair
Donaldson.

The code is released under the BSD license which can be found in
licenses/discovery_protocol

### LonestarGPU OpenCL

The original LonestarGPU code is released under the University of
Texas research license, which it maintains.

The LonestarGPU code is confined to the directory:
code/experiments/lonestar_gpu_opencl

A copy of the license can be found in:
licenses/lonestar_gpu

The original LonestarGPU code can be obtained from:
http://iss.ices.utexas.edu/?p=projects/galois/lonestargpu

### Pannotia

The Pannotia applications are provided with a license by AMD.

A copy of the license can be found in:
licenses/pannotia

The Pannotia code is confined to the directory:
code/experiments/pannotia

The original Pannotia code can be obtained from:
https://github.com/pannotia/pannotia

# Running the Code

### Obtaining Inputs

To run the Pannotia or LonestarGPU applications, you will need to obtain
the (large) graph files as input.

We provide instructions for obtaining these graphs in:
code/experiments/pannotia/datasets/README.txt
and 
code/experiments/lonestar_gpu_opencl/inputs/README.txt

We have provided google drive links containing these datasets.

### Running

We believe the code is straightforward to run. It uses a
cross-platform (Windows and Linux) CMake build system and the
applications give examples of the command line arguments if run with
no arguments.

Detailed instructions for running the code can be found in our OOPSLA
artifact evaluation guide: OOPSLA_AE.txt. This guide was reviewed and
accepted by the OOPSLA'16 artifact evaluation committee, so we believe
it should be sufficient.

# Known Issues

It was very difficult to get these applications running reliably across
all the GPUs we tested. We encountered many interesting issues, and we
even wrote a workshop paper about it that you should check out:

"The Hitchhiker's Guide to Cross-Platform OpenCL Application
Development" appearing in IWOCL'16

Here are some issues we have found that are still present in this code:

### Intel Compiler Crash

When running on Intel GPUs on Windows, the compiler seems to
non-deterministically crash (which crashes the application).

The application we find that suffers from this problem the worse
is the occupancy microbenchmarks

### LonestarGPU DMR Application

We have only been able to get this application working reliably on
Nvidia chips. And even on these chips, it crashes about 1% of the time
(same as the original CUDA application). Other chips deadlock, crash,
or produce wrong results 100% of the time.

The original DMR comes with a warning. We pass this warning remains
valid (and perhaps even more so) in our OpenCL port.

### Structs and Global Memory Pointers

Our LonestarGPU ports pass structures to kernels which contain global
memory pointers. This is technically undefined behaviour in OpenCL and
we know this to cause issues on ARM GPUs. However, it appears to work
fine on Nvidia, Intel and AMD GPUs.

This issue is fixable but requires a significant code refactoring.
Its on our todo list.
