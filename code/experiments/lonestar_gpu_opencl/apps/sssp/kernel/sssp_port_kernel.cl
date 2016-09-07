/** Single source shortest paths -*- C++ -*-
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
 * Single source shortest paths.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// sssp portable barrier kernel code (using the discovery
// protocol). Ported from the GPU-Lonestar sssp-worklistc application
// by Tyler Sorensen (2016)

typedef uint foru;
#define MYINFINITY 1000000000

#include "worklist_cl.h"
#include "graph_cl.h"
#include "discovery.cl"

#include "sssp_common.cl"

unsigned processnode2(__global foru *dist,
                      __global Graph *graph,
                      __global CL_Worklist2 *inwl,
                      __global CL_Worklist2 *outwl,
                      __local int* gather_offsets,
                      __local int* src,
                      __local int* scan_arr,
                      __local int* loc_tmp,
                      __local int* queue_index,
                      unsigned iteration,
                      __global discovery_kernel_ctx *gl_ctx,
                      __local  discovery_local_ctx *local_ctx
                      ) {

  int nn;
  uint id = p_get_global_id(gl_ctx, local_ctx);
  int threads = p_get_global_size(gl_ctx, local_ctx);

  int total_inputs = (*(inwl->dindex) +
                      threads - 1) /(threads);

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
            (scratch_offset + i - done) < WGS;
          i++) {

        gather_offsets[scratch_offset + i - done] = neighboroffset + neighborsdone + i;
        src[scratch_offset + i - done] = nn;
      }

      neighborsdone += i;
      scratch_offset += i;

      barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

      int ncnt = 0;
      unsigned to_push = 0;

      if (get_local_id(0) < total_edges) {

        if (processedge2(dist, graph, iteration, src[get_local_id(0)], gather_offsets[get_local_id(0)], &to_push)) {
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
            __global Graph *graph,
            __global CL_Worklist2 *inwl,
            __global CL_Worklist2 *outwl,
            __local int* gather_offsets,
            __local int* src,
            __local int* scan_arr,
            __local int* loc_tmp,
            __local int* queue_index,
            int iteration,
            __global discovery_kernel_ctx *gl_ctx,
            __local  discovery_local_ctx *local_ctx
            ) {

  unsigned id = p_get_global_id(gl_ctx, local_ctx);

  if (iteration == 0) {

    if (id == 0) {
      int item = 0;
      wl_push(inwl, item);
    }
    return;
  }
  else {
    processnode2(dist, graph, inwl, outwl, gather_offsets, src, scan_arr, loc_tmp, queue_index, iteration, gl_ctx, local_ctx);
  }
}

__kernel void drelax2(__global foru *dist,
                      __global Graph *graph,
                      __global CL_Worklist2 *inwl,
                      __global CL_Worklist2 *outwl,
                      int iteration,
                      __global discovery_kernel_ctx *gl_ctx
                      ) {

  // Entry point, wo we do the discovery protocol here.
  DISCOVERY_PROTOCOL(gl_ctx);
  __local int gather_offsets[WGS];
  __local int src[WGS];
  __local int scan_arr[WGS];
  __local int loc_tmp;
  __local int queue_index;

  if (iteration == 0) {
    drelax(dist, graph, inwl, outwl, gather_offsets, src, scan_arr, &loc_tmp, &queue_index, iteration, gl_ctx, &local_ctx);
  }
  else {
    __global CL_Worklist2 *in;
    __global CL_Worklist2 *out;
    __global CL_Worklist2 *tmp;

    in = inwl; out = outwl;

    while (*(in->dindex) > 0) {

      drelax(dist, graph, in, out, gather_offsets, src, scan_arr, &loc_tmp, &queue_index, iteration, gl_ctx, &local_ctx);

      // Inter-workgroup barrier
      discovery_barrier(gl_ctx, &local_ctx);

      tmp = in;
      in = out;
      out = tmp;

      *(out->dindex) = 0;

      iteration++;

      // Inter-workgroup barrier
      discovery_barrier(gl_ctx, &local_ctx);
    }
  }
}
//
