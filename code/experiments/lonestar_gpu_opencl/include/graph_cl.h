// The device part of the OpenCL CSR graph structure for GPU-Lonestar
// applications.

// OpenCL port of GPU-Lonestar applications by Tyler Sorensen (2016)

#pragma once

typedef struct {
  uint nnodes, nedges;
  __global uint *noutgoing, *nincoming, *srcsrc, *psrc, *edgessrcdst;
  __global uint *edgessrcwt;
  __global uint *maxOutDegree, *maxInDegree;

} Graph;

__kernel void init_graph(__global Graph *target,
                         uint nnodes,
                         uint nedges,
                         __global uint *noutgoing,
                         __global uint *nincoming,
                         __global uint *srcsrc,
                         __global uint *psrc,
                         __global uint *edgessrcdst,
                         __global uint *edgessrcwt,
                         __global uint *maxOutDegree,
                         __global uint *maxInDegree) {

  target->nnodes = nnodes;
  target->nedges = nedges;
  target->noutgoing = noutgoing;
  target->nincoming = nincoming;
  target->srcsrc = srcsrc;
  target->psrc = psrc;
  target->edgessrcdst = edgessrcdst;
  target->edgessrcwt = edgessrcwt;
  target->maxOutDegree = maxOutDegree;
  target->maxInDegree = maxInDegree;
}

unsigned g_getOutDegree(__global Graph *g, unsigned src) {
  return g->noutgoing[src];
}

unsigned g_getFirstEdge(__global Graph *g, unsigned src) {

  if (src < g->nnodes) {
    unsigned srcnout = g_getOutDegree(g, src);

    if (srcnout > 0 && g->srcsrc[src] < g->nnodes) {
      return g->psrc[g->srcsrc[src]];
    }
    return 0;
  }
  return 0;
}

unsigned g_getWeight(__global Graph *g, unsigned src, unsigned nthedge) {

  if (src < g->nnodes && nthedge < g_getOutDegree(g, src)) {
    unsigned edge = g_getFirstEdge(g, src) + nthedge;

    if (edge && edge < g->nedges + 1) {
      return g->edgessrcwt[edge];
    }
  }
  return MYINFINITY;
}

foru g_getDestination(__global Graph *g, unsigned src, unsigned nthedge) {

  if (src < g->nnodes && nthedge < g_getOutDegree(g, src)) {
    unsigned edge = g_getFirstEdge(g, src) + nthedge;

    if (edge && edge < g->nedges + 1) {
      return g->edgessrcdst[edge];
    }
  }
  return g->nnodes;
}
