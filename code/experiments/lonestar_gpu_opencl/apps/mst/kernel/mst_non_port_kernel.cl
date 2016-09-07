/** Minimum spanning tree -*- C++ -*-
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
 * @Description
 * Computes minimum spanning tree of a graph using Boruvka's algorithm.
 *
 * @author Rupesh Nasre <nasre@ices.utexas.edu>
 * @author Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// mst non-portable barrier kernel code (using the unsafe
// barrier). Ported from the GPU-Lonestar mst application by Tyler
// Sorensen (2016)

#include "mst_common.cl"

__kernel void dfindcompmintwo(__global unsigned *mstwt,                  // 0
                              __global Graph *graph,                     // 1
                              __global ComponentSpace *cs,               // 2
                              __global foru *eleminwts,                  // 3
                              __global foru *minwtcomponent,             // 4
                              __global unsigned *partners,               // 5
                              __global unsigned *phores,                 // 6
                              __global int *processinnextiteration,      // 7
                              __global unsigned *goaheadnodeofcomponent, // 8
                              __global int *repeat,                      // 9
                              __global int *count,                       // 10
                              __global atomic_int *gbar_arr              // 11
                              ) {

  unsigned tid = get_global_id(0);
  unsigned id, nthreads = get_global_size(0);
  unsigned up = (graph->nnodes + nthreads - 1) / nthreads * nthreads;
  unsigned srcboss, dstboss;

  for (id = tid; id < up; id += nthreads) {

    if (id < graph->nnodes && processinnextiteration[id]) {
        srcboss = cs_find(cs, id);
        dstboss = cs_find(cs, partners[id]);
    }

    // Unsafe global barrier
    gbar(gbar_arr);

    if (id < graph->nnodes && processinnextiteration[id] && srcboss != dstboss) {

      if (cs_unify(cs, srcboss, dstboss)) {
        atomic_add(mstwt, eleminwts[id]);
        atomic_add(count, 1);
        processinnextiteration[id] = false;
        eleminwts[id] = MYINFINITY; // Mark end of processing to avoid getting repeated.
      }
      else {
        *repeat = true;
      }
    }
    // Unsafe global barrier
    gbar(gbar_arr);
  }
}
//
