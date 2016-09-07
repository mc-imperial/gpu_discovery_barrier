// The device part of the OpenCL worklist structure for GPU-Lonestar
// applications.

// OpenCL port of GPU-Lonestar applications by Tyler Sorensen (2016)

#pragma once

#include "block_scan_cl.h"

typedef struct {

  __global int *dwl;
  __global int *dnsize;
  __global int *dindex;

} CL_Worklist2;

__kernel void init_worklist(__global CL_Worklist2 *wl,
                            __global int *dwl,
                            __global int *dnsize,
                            __global int *dindex) {
  wl->dwl = dwl;
  wl->dnsize = dnsize;
  wl->dindex = dindex;
}

int wl_push(__global CL_Worklist2 *wl, int ele) {

  int lindex = atomic_add(wl->dindex, 1);

  if (lindex >= *(wl->dnsize))
    return 0;

  wl->dwl[lindex] = ele;
  return 1;
}

int wl_pop_id(__global CL_Worklist2 *wl, int id, int *item) {

  if (id < *(wl->dindex)) {
    *item = wl->dwl[id];
    return 1;
  }
  return 0;
}

int wl_push_1item(__global CL_Worklist2 *wl, __local int *queue_index, __local int* scan_arr, __local int* loc_tmp, int nitem, int item, int threads_per_block){

  int total_items = 0;
  int thread_data = nitem;

  block_int_exclusive_sum_scan(scan_arr, loc_tmp, thread_data, &thread_data, &total_items, threads_per_block);

  if(get_local_id(0) == 0){
    *queue_index = atomic_add(wl->dindex, total_items);
  }

  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if(nitem == 1) {
    wl->dwl[*queue_index + thread_data] = item;
  }

  return total_items;
}
