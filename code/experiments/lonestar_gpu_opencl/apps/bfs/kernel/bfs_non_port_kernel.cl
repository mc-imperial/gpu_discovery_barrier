/** Breadth-first search -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in irregular
 * programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights reserved.
 * UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES CONCERNING THIS
 * SOFTWARE AND DOCUMENTATION, INCLUDING ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR ANY PARTICULAR PURPOSE, NON-INFRINGEMENT AND WARRANTIES OF
 * PERFORMANCE, AND ANY WARRANTY THAT MIGHT OTHERWISE ARISE FROM COURSE OF
 * DEALING OR USAGE OF TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH
 * RESPECT TO THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect, direct or
 * consequential damages or loss of profits, interruption of business, or
 * related expenses which may arise from use of Software or Documentation,
 * including but not limited to those resulting from defects in Software and/or
 * Documentation, or loss or inaccuracy of data of any kind.
 *
 * @section Description
 *
 * Example breadth-first search application for demoing Galois system.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// bfs non-portable barrier kernel code. Ported from the GPU-Lonestar
// bfs-worklistc application by Tyler Sorensen (2016)

#include "bfs_common.cl"
#include "gbar_cl.h"

uint processnode2(__global foru *dist,
                  __global Graph *graph,
                  __global CL_Worklist2 *inwl,
                  __global CL_Worklist2 *outwl,
                  __local int *gather_offsets,
                  __local int *queue_index,
                  __local int *scan_arr,
                  __local int *loc_tmp,
                  unsigned iteration
                  ) {

  const int SCRATCHSIZE = WGS;
  int nn;
  unsigned id = get_global_id(0);
  int threads = get_global_size(0);
  int total_inputs = (*(inwl->dindex) + threads - 1) / (threads);

  gather_offsets[get_local_id(0)] = 0;

  while (total_inputs-- > 0) {
    int neighborsize = 0;
    int neighboroffset = 0;
    int scratch_offset = 0;
    int total_edges = 0;

    if (wl_pop_id(inwl, id, &nn)) {
      if (nn != -1) {
        neighborsize = g_getOutDegree(graph, nn);
        neighboroffset = graph->psrc[graph->srcsrc[nn]];
      }
    }

    block_int_exclusive_sum_scan(scan_arr, loc_tmp, neighborsize, &scratch_offset, &total_edges, WGS);

    int done = 0;
    int neighborsdone = 0;

    while(total_edges > 0) {
      int i;
      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      for(i = 0;
          neighborsdone + i < neighborsize &&
            (scratch_offset + i - done) < SCRATCHSIZE;
          i++
          ) {
        gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
      }

      neighborsdone += i;
      scratch_offset += i;

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      int ncnt = 0;
      unsigned to_push = 0;

      if (get_local_id(0) < total_edges) {

        if (processedge2(dist, graph, iteration, gather_offsets[get_local_id(0)], &to_push)) {
          ncnt = 1;
        }
      }

      wl_push_1item(outwl, queue_index, scan_arr, loc_tmp, ncnt, (int) to_push, WGS);

      total_edges -= WGS;
      done += WGS;
    }

    id += threads;
  }
  return 0;
}


void drelax(__global foru *dist,
            __global Graph* graph,
            __global uint *gerrno,
            __global CL_Worklist2 *inwl,
            __global CL_Worklist2 *outwl,
            __local int* gather_offsets,
            __local int* queue_index,
            __local int* scan_arr,
            __local int* loc_tmp,
            int iteration
            ) {
  unsigned id = get_global_id(0);

  if (iteration == 0) {

    if (id == 0) {
      int item = 0;
      wl_push(inwl, item);
    }
    return;
  }
  else {
    if (processnode2(dist, graph, inwl, outwl, gather_offsets, queue_index, scan_arr, loc_tmp, iteration)) {
      *gerrno = 1;
    }
  }
}

__kernel void drelax2(__global foru *dist,
                      __global Graph *graph,
                      __global uint *gerrno,
                      __global CL_Worklist2 *inwl,
                      __global CL_Worklist2 *outwl,
                      int iteration,
                      __global atomic_int* gbar_arr
                      ) {

  __local int gather_offsets[WGS];
  __local int queue_index;
  __local int scan_arr[WGS];
  __local int loc_tmp;

  // Special case if iteration == 1
  if (iteration == 0) {
    drelax(dist, graph, gerrno, inwl, outwl, gather_offsets, &queue_index, scan_arr, &loc_tmp, iteration);
  }
  else {
    __global CL_Worklist2 *in;
    __global CL_Worklist2 *out;
    __global CL_Worklist2 *tmp;

    in = inwl; out = outwl;

    while (*(in->dindex) > 0) {

      drelax(dist, graph, gerrno, in, out, gather_offsets, &queue_index, scan_arr, &loc_tmp, iteration);

      gbar(gbar_arr);

      tmp = in;
      in = out;
      out = tmp;

      *(out->dindex) = 0;

      iteration++;

      // Tyler: Added to avoid a data-race present in the original code
      gbar(gbar_arr);
    }
  }

}
//