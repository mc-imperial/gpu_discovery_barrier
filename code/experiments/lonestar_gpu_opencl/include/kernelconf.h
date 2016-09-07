// A simple kernel configuration file ported to OpenCL for the
// GPU-Lonestar applications. The structure is able to take a problem
// size and the threads per workgroups and compute how many workgroups
// are required.

// OpenCL port by Tyler Sorensen (2016)

#pragma once

typedef struct {
  unsigned problemsize;
  unsigned nwgs, wgs, gs;
} KernelConfig;


void set_problem_size(KernelConfig *k, unsigned ps) {
  k->problemsize = ps;
}

void set_workgroup_size(KernelConfig *k, unsigned wgs) {
  k->wgs = wgs;
}

void calculate_kernelconf(KernelConfig *k) {
  k->nwgs = (k->problemsize + k->wgs - 1)/k->wgs;
  k->gs = k->nwgs * k->wgs;
}
