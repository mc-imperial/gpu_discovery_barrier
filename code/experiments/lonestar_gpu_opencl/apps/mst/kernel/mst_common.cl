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

// Common functions between the portable and non-portable mst application.

// OpenCL port of GPU-Lonestar mst application by Tyler Sorensen (2016)

typedef uint foru;
#define MYINFINITY 1000000000

#include "graph_cl.h"
#include "component_cl.h"
#include "discovery.cl"
#include "gbar_cl.h"

// Kernel it initialise device side buffers
__kernel void dinit(__global unsigned *mstwt,             // 0
                    __global Graph *graph,                // 1
                    __global ComponentSpace *cs,          // 2
                    __global foru *eleminwts,             // 3
                    __global foru *minwtcomponent,        // 4
                    __global unsigned *partners,          // 5
                    __global unsigned *phores,            // 6
                    __global int *processinnextiteration, // 7
                    __global unsigned *goaheadnodeofcomponent) { // 8

  int id = get_global_id(0);

  if (id < graph->nnodes) {
    eleminwts[id] = MYINFINITY;
    minwtcomponent[id] = MYINFINITY;
    goaheadnodeofcomponent[id] = graph->nnodes;
    phores[id] = 0;
    partners[id] = id;
    processinnextiteration[id] = 0;
  }
}


__kernel void dfindelemin(__global unsigned *mstwt,             // 0
                          __global Graph *graph,                // 1
                          __global ComponentSpace *cs,          // 2
                          __global foru *eleminwts,             // 3
                          __global foru *minwtcomponent,        // 4
                          __global unsigned *partners,          // 5
                          __global unsigned *phores,            // 6
                          __global int *processinnextiteration, // 7
                          __global unsigned *goaheadnodeofcomponent) { // 8

  int id = get_global_id(0);

  if (id < graph->nnodes) {

    // If I have a cross-component edge, find my minimum wt
    // cross-component edge, inform my boss about this edge
    // (atomicMin).

    unsigned src = id;
    unsigned srcboss = cs_find(cs, src);
    unsigned dstboss = graph->nnodes;
    foru minwt = MYINFINITY;
    unsigned degree = g_getOutDegree(graph, src);

    for (unsigned ii = 0; ii < degree; ++ii) {
      foru wt = g_getWeight(graph, src, ii);

      if (wt < minwt) {
        unsigned dst = g_getDestination(graph, src, ii);
        unsigned tempdstboss = cs_find(cs, dst);

        if (srcboss != tempdstboss) { // Cross-component edge.
          minwt = wt;
          dstboss = tempdstboss;
        }
      }
    }
    eleminwts[id] = minwt;
    partners[id] = dstboss;

    if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {

      // Inform boss.
      foru oldminwt = atomic_min(&(minwtcomponent[srcboss]), minwt);
    }
  }
}

__kernel void dfindelemin2(__global unsigned *mstwt,             // 0
                           __global Graph *graph,                // 1
                           __global ComponentSpace *cs,          // 2
                           __global foru *eleminwts,             // 3
                           __global foru *minwtcomponent,        // 4
                           __global unsigned *partners,          // 5
                           __global unsigned *phores,            // 6
                           __global int *processinnextiteration, // 7
                           __global unsigned *goaheadnodeofcomponent) { // 8

  int id = get_global_id(0);

  if (id < graph->nnodes) {
    unsigned src = id;
    unsigned srcboss = cs_find(cs, src);

    if(eleminwts[id] == minwtcomponent[srcboss] && srcboss != partners[id] && partners[id] != graph->nnodes) {
      unsigned degree = g_getOutDegree(graph, src);

      for (unsigned ii = 0; ii < degree; ++ii) {
        foru wt = g_getWeight(graph, src, ii);

        if (wt == eleminwts[id]) {
          unsigned dst = g_getDestination(graph, src, ii);
          unsigned tempdstboss = cs_find(cs, dst);

          if (tempdstboss == partners[id]) { // Cross-component edge.
            atomic_min(&goaheadnodeofcomponent[srcboss], id);
          }
        }
      }
    }
  }
}

__kernel void verify_min_elem(__global unsigned *mstwt,             // 0
                              __global Graph *graph,                // 1
                              __global ComponentSpace *cs,          // 2
                              __global foru *eleminwts,             // 3
                              __global foru *minwtcomponent,        // 4
                              __global unsigned *partners,          // 5
                              __global unsigned *phores,            // 6
                              __global int *processinnextiteration, // 7
                              __global unsigned *goaheadnodeofcomponent) { // 8

  int id = get_global_id(0);

  if (id < graph->nnodes) {

    if(cs_isBoss(cs, id)) {

      if(goaheadnodeofcomponent[id] == graph->nnodes) {
        return;
      }

      unsigned minwt_node = goaheadnodeofcomponent[id];

      unsigned degree = g_getOutDegree(graph, minwt_node);
      foru minwt = minwtcomponent[id];

      if(minwt == MYINFINITY) {
        return;
      }

      bool minwt_found = false;

      for (unsigned ii = 0; ii < degree; ++ii) {
        foru wt = g_getWeight(graph, minwt_node, ii);

        if (wt == minwt) {
          minwt_found = true;
          unsigned dst = g_getDestination(graph, minwt_node, ii);
          unsigned tempdstboss = cs_find(cs, dst);

          if(tempdstboss == partners[minwt_node] && tempdstboss != id) {
            processinnextiteration[minwt_node] = true;
            return;
          }
        }
      }
    }
  }
}
