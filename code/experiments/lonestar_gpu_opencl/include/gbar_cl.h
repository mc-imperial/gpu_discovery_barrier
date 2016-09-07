// The device part of an OpenCL port the Xaio/Feng GPU global barrier
// using OpenCL 2.0 atomics. This is an unsafe implementation. That
// is, it attempts to synchronise across all workgroups on the GPU.

// A safe variant (using the discovery protocol) is available in the
// discovery_protocol directory

// Port by Tyler Sorensen (2016)

#pragma once

// Call this function to synchronise across all workgroups.  May
// deadlock if the kernel is launched with more workgroups than the
// occupancy of the GPU.
void gbar(__global atomic_int *gbar_arr) {

  int id = get_group_id(0);

  // This barrier actually isn't needed but some GPUs crash if it
  // isn't included (!!)
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  // Master workgroup goes here
  if (id == 0) {

    // Each thread in master is responsible for distinct slave(s)
    for (int peer_block = get_local_id(0) + 1;
         peer_block < get_num_groups(0);
         peer_block += get_local_size(0)) {

      // Wait for the slave
      while (atomic_load_explicit(&(gbar_arr[peer_block]), memory_order_relaxed, memory_scope_device) == 0);

      // Synchronise
      atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, memory_order_acquire, memory_scope_device);
    }

    // Wait for all slaves
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    for (int peer_block = get_local_id(0) + 1; peer_block < get_num_groups(0); peer_block += get_local_size(0)) {

      // Release slaves
      atomic_store_explicit(&(gbar_arr[peer_block]), 0, memory_order_release, memory_scope_device);
    }
  }

  // Slave workgroups go here
  else {

    // All threads per slave sync here
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // One rep per slave
    if (get_local_id(0) == 0) {

      // Mark arrival
      atomic_store_explicit(&(gbar_arr[id]), 1, memory_order_release, memory_scope_device);

      // Wait to be released by the master
      while (atomic_load_explicit(&(gbar_arr[id]), memory_order_relaxed, memory_scope_device) == 1);

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
