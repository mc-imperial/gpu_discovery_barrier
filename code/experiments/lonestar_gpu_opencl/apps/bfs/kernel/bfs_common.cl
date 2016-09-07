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

// Common functions for both portable and non-portable bfs

// Ported from the GPU-Lonestar bfs-worklistc application by Tyler
// Sorensen (2016)

typedef uint foru;
#define MYINFINITY 1000000000
#include "graph_cl.h"

#include "discovery.cl"
#include "worklist_cl.h"
#include "block_scan_cl.h"

// Kernel to validate the solution
__kernel void dverifysolution(__global foru *dist, __global Graph *graph, __global uint *nerr) {

  unsigned int nn = get_global_id(0);

  if (nn < graph->nnodes) {
    uint nsrcedges = g_getOutDegree(graph, nn);

    for (unsigned ii = 0; ii < nsrcedges; ++ii) {
      unsigned int u = nn;
      unsigned int v = g_getDestination(graph, u, ii);
      foru wt = 1;

      if (wt > 0 && dist[u] + wt < dist[v]) {
        atomic_add(nerr, 1);
      }
    }
  }
}


// Kernel to initialise the distance array
__kernel void initialize(__global foru *dist, uint nv) {

  unsigned int ii = get_global_id(0);

  if (ii < nv) {
    dist[ii] = MYINFINITY;
  }
}

foru processedge2(__global foru *dist,
                  __global Graph *graph,
                  uint iteration,
                  uint edge,
                  uint *dst) {

  *dst = graph->edgessrcdst[edge];
  if (*dst >= graph->nnodes) return 0;

  foru wt = 1;
  if (wt >= MYINFINITY) return 0;

  if(dist[*dst] == MYINFINITY) {
    dist[*dst] = iteration;
    return MYINFINITY;
  }
  return 0;
}
