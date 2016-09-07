// Header file to use in the kernel for using the discovery protocol.
// Contains the actual discovery protocol and barrier functions

#pragma once

#ifdef CUSTOM_ATOMICS
#include "../custom_atomics/custom_atomics.cl"
#endif

#include "common.h"

#include "../locks/locks.cl"

// The maximum number of participating groups
#define BAR_FLAG_SIZE 1024

// Discovery execution environment variables can be stored in local
// memory (discovery_local_ctx)
typedef struct {
  INT_TYPE participating_group_id;
  INT_TYPE participating_group_size;
  INT_TYPE is_participating;
} discovery_local_ctx;

/*
  Functions for Discovery protocol, execution environment, and XF barrier.
*/

// Reset the discovery kernel ctx so that it can be used in
// subsequent kernels without explicit reset.
void reset_kernel_context(__global discovery_kernel_ctx *gl_ctx) {
  gl_ctx->prot_poll_open = 1;
  gl_ctx->prot_counter = 0;
  atomic_store_explicit(&(gl_ctx->m.counter), 0, memory_order_relaxed, memory_scope_device);
  gl_ctx->kernel_counter = 0;
  atomic_store_explicit(&(gl_ctx->m.now_serving), 0, memory_order_relaxed, memory_scope_device);
}

// Discovery protocol that is executed by one representative thread per workgroup
// Has functionality to reset after every use (not discussed in OOPSLA paper).
// This allows the protocol to be used in succession without any explicit reset.
void discovery_protocol_master(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {

  int reset_gl_memory = 0;
  int total_work_groups = get_num_groups(0);

  // Polling phase
  discovery_lock(&(gl_ctx->m));

  if (gl_ctx->prot_poll_open) { // Poll is open
    int id = gl_ctx->prot_counter;
    local_ctx->is_participating = 1;
    local_ctx->participating_group_id = id;

    gl_ctx->prot_counter++;
    discovery_unlock(&(gl_ctx->m));
  }
  else { // Poll is closed
    local_ctx->is_participating = 0;
    discovery_unlock(&(gl_ctx->m));
  }

  // Closing phase
  discovery_lock(&(gl_ctx->m));
  gl_ctx->kernel_counter++;

  // Last workgroup through resets the protocol so that
  // it may be used in a subsequent kernel without reset
  if (gl_ctx->kernel_counter == total_work_groups) {
    reset_gl_memory = 1;
  }

  if (gl_ctx->prot_poll_open) { // Poll open
    gl_ctx->prot_poll_open = 0;
    gl_ctx->num_participating = gl_ctx->prot_counter;
  }

  local_ctx->participating_group_size = gl_ctx->prot_counter;

  if (reset_gl_memory) {

    // Contains implicit unlock
    reset_kernel_context(gl_ctx);
  }
  else {
    discovery_unlock(&(gl_ctx->m));
  }
}

// Top level discovery protocol function
void discovery_protocol(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {

  // Only 1 representative thread per workgroup
  int id_flag = get_local_id(0);

  if (id_flag == 0) {
    discovery_protocol_master(gl_ctx, local_ctx);
  }

  // All other threads in the workgroup wait here for the result
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

// Returns 1 if the workgroup was one of the occupant workgroups
// discovered by the discovery protocol (participating workgroups).
int is_participating(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {
  return local_ctx->is_participating;
}

// Occupancy discovery execution environment functions
// as described in paper
int p_get_num_groups(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {
  return local_ctx->participating_group_size;
}

int p_get_group_id(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {
  return local_ctx->participating_group_id;
}

int p_get_global_id(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {
  return get_local_id(0) + get_local_size(0) * p_get_group_id(gl_ctx, local_ctx);
}

int p_get_global_size(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {
  return p_get_num_groups(gl_ctx, local_ctx) * get_local_size(0);
}

// An implementation of the XF barrier
void XF_barrier(__global discovery_kernel_ctx *gl_ctx, __local discovery_local_ctx *local_ctx) {

  int id = p_get_group_id(gl_ctx, local_ctx);

  // This barrier actually isn't needed but some GPUs crash if it
  // isn't included (!!)
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // Master workgroup goes here
  if (id == 0) {

    // Each thread in master is responsible for distinct slave(s)
    for (int peer_block = get_local_id(0) + 1;
         peer_block < p_get_num_groups(gl_ctx, local_ctx);
         peer_block += get_local_size(0)) {

      // Wait for the slave
      while (atomic_load_explicit(&(gl_ctx->bar_flags[peer_block]), memory_order_relaxed, memory_scope_device) == 0);

      // Synchronise
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }

    // Wait for all slaves
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int peer_block = get_local_id(0) + 1;
         peer_block < p_get_num_groups(gl_ctx, local_ctx);
         peer_block += get_local_size(0)) {

      // Release slaves
      atomic_store_explicit(&(gl_ctx->bar_flags[peer_block]), 0, memory_order_release, memory_scope_device);
    }
  }

  // Slave workgroups go here
  else {

    // All threads per slave sync here
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // One rep per slave
    if (get_local_id(0) == 0) {

      // Mark arrival
      atomic_store_explicit(&(gl_ctx->bar_flags[id]), 1, memory_order_release, memory_scope_device);

      // Wait to be released by the master
      while (atomic_load_explicit(&(gl_ctx->bar_flags[id]), memory_order_relaxed, memory_scope_device) == 1);

      // Synchronise
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }

    // All threads per slave are released here
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }

  // This barrier actually isn't needed but some GPUs crash if it
  // isn't included (!!)
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

// Explicitly initialise the discovery_kernel_ctx. Should only be
// needed for the very first time the protocol is used.
__kernel void init_discovery_kernel_ctx(__global discovery_kernel_ctx *gl_ctx, int skip) {

  reset_kernel_context(gl_ctx);
  gl_ctx->skip = skip;

  for (int i = 0; i < BAR_FLAG_SIZE; i++) {
    atomic_store_explicit(&(gl_ctx->bar_flags[i]), 0, memory_order_relaxed, memory_scope_device);
  }
}

/*
  Macros for the discovery protocol and barrier
 */

// Use #define in case another barrier wants to be experimented with
#define discovery_barrier(gl_ctx, local_ctx) XF_barrier(gl_ctx, local_ctx)

// The high level protocol that runs the discovery protocol and
// forces non participating groups to exit.
#define DISCOVERY_PROTOCOL(gl_ctx)                                      \
  __local discovery_local_ctx local_ctx;                                \
  if (gl_ctx->skip) {                                                   \
    local_ctx.participating_group_size = get_num_groups(0);             \
    local_ctx.is_participating = 1;                                     \
    local_ctx.participating_group_id = get_group_id(0);                 \
    gl_ctx->num_participating = get_num_groups(0);                      \
  }                                                                     \
  else {                                                                \
    discovery_protocol(gl_ctx, &local_ctx);                             \
    if (!is_participating(gl_ctx, &local_ctx)) {                        \
      return;                                                           \
    }                                                                   \
  }                                                                     \
  
