// Data structures that are common both to the host and device.

#pragma once

#ifdef CUSTOM_ATOMICS
#include "../custom_atomics/custom_atomics.cl"
#endif

// For kernel code, define INT_TYPE as int, for host code cl_int
#ifndef INT_TYPE
#error "INT_TYPE not defined"
#endif

// For kernel code, define ATOMIC_INT_TYPE as atomic_int, for host code cl_int
#ifndef ATOMIC_INT_TYPE
#error "ATOMIC_INT_TYPE not defined"
#endif

// The maximum number of participating groups
#define BAR_FLAG_SIZE 1024

/*
  Mutex, discovery protocol, execution environment, and XF barrier data structures
*/

// Mutex
typedef struct {
  ATOMIC_INT_TYPE counter;
  ATOMIC_INT_TYPE now_serving;
} discovery_mutex;

// Barrier and discovery protocol variables (discovery_kernel_ctx)
typedef struct {

  // Barrier flags for XF Barrier
  ATOMIC_INT_TYPE bar_flags[BAR_FLAG_SIZE];

  // Discovery protocol variables
  INT_TYPE prot_poll_open;
  INT_TYPE prot_counter;
  INT_TYPE num_participating;
  INT_TYPE kernel_counter;
  discovery_mutex m;

  // Flag to skip the protocol for finding the occupancy bound
  INT_TYPE skip;

} discovery_kernel_ctx;
