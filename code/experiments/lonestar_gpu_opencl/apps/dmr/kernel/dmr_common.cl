/** Delaunay refinement -*- C++ -*-
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
 * Refinement of an initial, unrefined Delaunay mesh to eliminate triangles
 * with angles < 30 degrees
 *
 * @author: Sreepathi Pai <sreepai@ices.utexas.edu>
 */

// The GPU-Lonestar dmr application ported to OpenCL.  Port by Tyler
// Sorensen (2016)

// These are the common device functions between the portable and
// non-portable dmr variants

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define FORD double
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#define FORD double
#else
#error "Double precision floating point not supported by OpenCL implementation."
#endif

#include "worklist_cl.h"
#include "discovery.cl"
#include "gbar_cl.h"

#define INVALIDID 1234567890
#define MAXID INVALIDID

#define MINANGLE 30

#define PI 3.14159265358979323846 // From C99 standard.

#define IS_SEGMENT(element) (((element).z == INVALIDID))

#define CAVLEN 256
#define BCLEN 1024

typedef struct {
  uint maxnelements;
  uint maxnnodes;
  uint ntriangles;
  uint nnodes;
  uint nsegments;
  uint nelements;

  __global FORD *nodex;
  __global FORD *nodey;
  __global uint3 *elements;
  __global int *isdel;
  __global int *isbad;
  __global uint3 *neighbours;
  __global int *owners;

} Mesh;

__kernel void init_mesh(__global Mesh *m,
                        __global FORD *nodex,
                        __global FORD *nodey,
                        __global uint3 *elements,
                        __global uint3 *neighbours,
                        __global int *isdel,
                        __global int *isbad,
                        __global int *owners,
                        uint maxnelements,
                        uint maxnnodes,
                        uint ntriangles,
                        uint nnodes,
                        uint nsegments,
                        uint nelements
                        ) {
  m->nodex = nodex;
  m->nodey = nodey;
  m->elements = elements;
  m->isdel = isdel;
  m->isbad = isbad;
  m->neighbours = neighbours;
  m->owners = owners;
  m->maxnelements = maxnelements;
  m->ntriangles = ntriangles;
  m->nnodes = nnodes;
  m->nsegments = nsegments;
  m->nelements = nelements;
}

FORD distanceSquare2(FORD onex, FORD oney, FORD twox, FORD twoy) {
  FORD dx = onex - twox;
  FORD dy = oney - twoy;
  FORD dsq = dx * dx + dy * dy;
  return dsq;
}


FORD distanceSquare(unsigned one, unsigned two, __global FORD *nodex, __global FORD *nodey) {
  return distanceSquare2(nodex[one], nodey[one], nodex[two], nodey[two]);
}

int angleLT(__global Mesh *mesh, unsigned a, unsigned b, unsigned c) {

  FORD vax = mesh->nodex[a] - mesh->nodex[c];
  FORD vay = mesh->nodey[a] - mesh->nodey[c];
  FORD vbx = mesh->nodex[b] - mesh->nodex[c];
  FORD vby = mesh->nodey[b] - mesh->nodey[c];
  FORD dp = vax * vbx + vay * vby; // Dot product

  if (dp < 0.0) {

    // ID is obtuse at point ii.
    return 0;
  }  else {
    FORD dsqaacurr = distanceSquare(a, c, mesh->nodex, mesh->nodey);
    FORD dsqbbcurr = distanceSquare(b, c, mesh->nodex, mesh->nodey);
    FORD c = dp * rsqrt(dsqaacurr * dsqbbcurr);
    if (c > cos(MINANGLE * (PI / 180))) {
      return 1;
    }
  }
  return 0;
}


__kernel void check_triangles(__global Mesh *mesh,
                              __global uint *bad_triangles,
                              __global CL_Worklist2 *wl,
                              int start) {

  __global uint3 *el;
  int id = get_global_id(0);
  int threads = get_global_size(0);
  int ele, push, count = 0, ulimit = mesh->nelements;

  for (ele = id + start; ele < ulimit; ele += threads) {
    push = 0;

    if (ele < mesh->nelements) {

      if (mesh->isdel[ele]) {
        goto next;
      }

      if(IS_SEGMENT(mesh->elements[ele])) {
        goto next;
      }

      if (!mesh->isbad[ele]) {

        el = &(mesh->elements[ele]);

        mesh->isbad[ele] = (angleLT(mesh, el->x, el->y, el->z)
                            || angleLT(mesh, el->z, el->x, el->y)
                            || angleLT(mesh, el->y, el->z, el->x));
      }

      if (mesh->isbad[ele]) {
        push = 1;
        count++;
      }
    }

  next:
    if (push) wl_push(wl, ele);
  }

  atomic_add(bad_triangles, count);
}

int angleOB(__global Mesh *mesh, unsigned a, unsigned b, unsigned c) {
  FORD vax = mesh->nodex[a] - mesh->nodex[c];
  FORD vay = mesh->nodey[a] - mesh->nodey[c];
  FORD vbx = mesh->nodex[b] - mesh->nodex[c];
  FORD vby = mesh->nodey[b] - mesh->nodey[c];
  FORD dp = vax * vbx + vay * vby; // Dot Product

  if (dp < 0.0)
    return 1;

  return 0;
}

void find_shared_edge(const uint3 elem1, const uint3 elem2, uint *se) {
  int sc = 0;
  if (elem1.x == elem2.x || elem1.x == elem2.y || elem1.x == elem2.z)
    se[sc++] = elem1.x;

  if (elem1.y == elem2.x || elem1.y == elem2.y || elem1.y == elem2.z)
    se[sc++] = elem1.y;

  if (!IS_SEGMENT(elem1) && (elem1.z == elem2.x || elem1.z == elem2.y || elem1.z == elem2.z))
    se[sc++] = elem1.z;
}

uint opposite(__global Mesh *mesh, uint element) {
  bool obtuse = 0;
  int obNode = INVALIDID;
  uint3 el = mesh->elements[element];

  if (IS_SEGMENT(el))
    return element;

  // Figure out the obtuse node
  if (angleOB(mesh, el.x, el.y, el.z)) {
    obtuse = 1;
    obNode = el.z;
  }
  else {

    if(angleOB(mesh, el.z, el.x, el.y)) {
      obtuse = 1;
      obNode = el.y;
    }
    else {

      if(angleOB(mesh, el.y, el.z, el.x)) {
        obtuse = 1;
        obNode = el.x;
      }
    }
  }

  if (obtuse) {

    // Find the neighbour that shares an edge whose points do not
    // include obNode
    uint se_nodes[2];
    uint nobneigh;

    uint3 neigh = mesh->neighbours[element];
    nobneigh = neigh.x;
    find_shared_edge(el, mesh->elements[neigh.x], se_nodes);

    if(se_nodes[0] == obNode || se_nodes[1] == obNode) {
      nobneigh = neigh.y;
      find_shared_edge(el, mesh->elements[neigh.y], se_nodes);

      if(se_nodes[0] == obNode || se_nodes[1] == obNode) {
        nobneigh = neigh.z;
      }
    }
    return nobneigh;
  }
  return element;
}

void circumcenter(FORD Ax,
                  FORD Ay,
                  FORD Bx,
                  FORD By,
                  FORD Cx,
                  FORD Cy,
                  FORD *CCx,
                  FORD *CCy) {
  FORD D;

  D = 2 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By));

  *CCx = ((Ax*Ax + Ay*Ay)*(By - Cy) +
          (Bx*Bx + By*By)*(Cy - Ay) +
          (Cx*Cx + Cy*Cy)*(Ay - By))/D;

  *CCy = ((Ax*Ax + Ay*Ay)*(Cx - Bx) +
          (Bx*Bx + By*By)*(Ax - Cx) +
          (Cx*Cx + Cy*Cy)*(Bx - Ax))/D;
}

FORD counterclockwise(FORD pax, FORD pay, FORD pbx, FORD pby, FORD pcx, FORD pcy) {

  FORD detleft, detright, det;

  detleft = (pax - pcx) * (pby - pcy);
  detright = (pay - pcy) * (pbx - pcx);
  det = detleft - detright;

  return det;
}


FORD gincircle (FORD ax, FORD ay, FORD bx, FORD by, FORD cx, FORD cy, FORD px, FORD py) {

  FORD apx, bpx, cpx, apy, bpy, cpy;
  FORD bpxcpy, cpxbpy, cpxapy, apxcpy, apxbpy, bpxapy;
  FORD alift, blift, clift, det;

  apx = ax - px;
  bpx = bx - px;
  cpx = cx - px;

  apy = ay - py;
  bpy = by - py;
  cpy = cy - py;

  bpxcpy = bpx * cpy;
  cpxbpy = cpx * bpy;
  alift = apx * apx + apy * apy;

  cpxapy = cpx * apy;
  apxcpy = apx * cpy;
  blift = bpx * bpx + bpy * bpy;

  apxbpy = apx * bpy;
  bpxapy = bpx * apy;
  clift = cpx * cpx + cpy * cpy;

  det = alift * (bpxcpy - cpxbpy) + blift * (cpxapy - apxcpy) + clift * (apxbpy - bpxapy);

  return det;
}

int encroached(__global Mesh *mesh, int element, uint3 *celement, FORD centerx, FORD centery, int *is_seg) {

  if(element == INVALIDID)
    return 0;

  uint3 ele = mesh->elements[element];

  if (IS_SEGMENT(ele)) {

    FORD cx, cy, radsqr;
    uint nsp;

    *is_seg = 1;

    nsp = (celement->x == ele.x) ? ((celement->y == ele.y) ? celement->z : celement->y) : celement->x;

    // Check if center and triangle are on opposite sides of segment
    // One of the ccws does not return zero
    if(counterclockwise(mesh->nodex[ele.x], mesh->nodey[ele.x],
                        mesh->nodex[ele.y], mesh->nodey[ele.y],
                        mesh->nodex[nsp], mesh->nodey[nsp]) > 0.0 !=
       counterclockwise(mesh->nodex[ele.x], mesh->nodey[ele.x],
                        mesh->nodex[ele.y], mesh->nodey[ele.y],
                        centerx, centery) > 0.0)
      return 1;

    // Nope, do a distance check
    cx = (mesh->nodex[ele.x] + mesh->nodex[ele.y]) / 2;
    cy = (mesh->nodey[ele.x] + mesh->nodey[ele.y]) / 2;

    // Replaced with OpenCL native distance functions
    radsqr = distanceSquare2(cx, cy, mesh->nodex[ele.x], mesh->nodey[ele.x]);

    return distanceSquare2(centerx, centery, cx, cy) < radsqr;
  }
  else
    return gincircle(mesh->nodex[ele.x], mesh->nodey[ele.x],
                     mesh->nodex[ele.y], mesh->nodey[ele.y],
                     mesh->nodex[ele.z], mesh->nodey[ele.z],
                     centerx, centery) > 0.0;
}

void add_to_cavity(uint *cavity, uint *cavlen, int element) {
  int i;

  for (i = 0; i < *cavlen; i++)

    if (cavity[i] == element)
      return;

  // Original code uses '++' which OpenCL doesn't like
  // Original: cavity[cavlen++] = element;
  cavity[*cavlen] = element;
  *cavlen = *cavlen + 1;
}

void add_to_boundary(uint *boundary, uint *boundarylen, uint sn1, uint sn2, uint src, uint dst) {

  int i;
  for(i = 0; i < *boundarylen; i+=4)

    if((sn1 == boundary[i] && sn2 == boundary[i+1]) ||
       (sn1 == boundary[i+1] && sn2 == boundary[i]))
      return;

  // A bunch of operations that originally used the '++' operator
  boundary[*boundarylen] = sn1;
  *boundarylen = *boundarylen + 1;
  boundary[*boundarylen] = sn2;
  *boundarylen = *boundarylen + 1;
  boundary[*boundarylen] = src;
  *boundarylen = *boundarylen + 1;
  boundary[*boundarylen] = dst;
  *boundarylen = *boundarylen + 1;
}


int build_cavity(__global Mesh *mesh,
                 uint *cavity,
                 uint *cavlen,
                 int max_cavity,
                 uint *boundary,
                 uint *boundarylen,
                 FORD *cx, FORD *cy) {

  int ce = 0;
  uint3 ele = mesh->elements[cavity[0]];
  int is_seg = 0;

  if (IS_SEGMENT(ele)) {
    *cx = (mesh->nodex[ele.x] + mesh->nodex[ele.y]) / 2;
    *cy = (mesh->nodey[ele.x] + mesh->nodey[ele.y]) / 2;
  }
  else {
    circumcenter(mesh->nodex[ele.x], mesh->nodey[ele.x],
                 mesh->nodex[ele.y], mesh->nodey[ele.y],
                 mesh->nodex[ele.z], mesh->nodey[ele.z],
                 cx, cy);
  }

  while (ce < *cavlen) {

    uint3 neighbours = mesh->neighbours[cavity[ce]];
    uint neighb[3] = {neighbours.x, neighbours.y, neighbours.z};

    for (int i = 0; i < 3; i++) {
      if (neighb[i] == cavity[0])
        continue;

      if(neighb[i] == INVALIDID)
        continue;

      is_seg  = 0;
      if (!(IS_SEGMENT(ele) && IS_SEGMENT(mesh->elements[neighb[i]])) &&
         encroached(mesh, neighb[i], &ele, *cx, *cy, &is_seg)) {
        if (!is_seg)
          add_to_cavity(cavity, cavlen, neighb[i]);
        else {
          cavity[0] = neighb[i];
          *cavlen = 1;
          *boundarylen = 0;
          return 0;
        }
      }
      else {
        uint se[2];
        find_shared_edge(mesh->elements[cavity[ce]], mesh->elements[neighb[i]], se);
        add_to_boundary(boundary, boundarylen, se[0], se[1], neighb[i], cavity[ce]);
      }
    }
    ce++;
  }
  return 1;
}

unsigned add_node(__global Mesh *mesh, FORD x, FORD y, uint ndx) {

  mesh->nodex[ndx] = x;
  mesh->nodey[ndx] = y;

  return ndx;
}

uint add_segment(global Mesh *mesh, uint n1, uint n2, uint ndx) {
  uint3 ele;
  ele.x = n1; ele.y = n2; ele.z = INVALIDID;

  mesh->isbad[ndx] = 0;
  mesh->isdel[ndx] = 0;
  mesh->elements[ndx] = ele;
  mesh->neighbours[ndx].x = mesh->neighbours[ndx].y = mesh->neighbours[ndx].z = INVALIDID;

  return ndx;
}

uint add_triangle(__global Mesh *mesh, uint n1, uint n2, uint n3, uint nb1, uint oldt, uint ndx) {

  uint3 ele;
  if (counterclockwise(mesh->nodex[n1], mesh->nodey[n1],
                      mesh->nodex[n2], mesh->nodey[n2],
                      mesh->nodex[n3], mesh->nodey[n3]) > 0) {
    ele.x = n1; ele.y = n2; ele.z = n3;
  }
  else {
    ele.x = n3; ele.y = n2; ele.z = n1;
  }

  mesh->isbad[ndx] = 0;
  mesh->isdel[ndx] = 0;
  mesh->elements[ndx] = ele;
  mesh->neighbours[ndx].x = nb1;

  mesh->neighbours[ndx].y = mesh->neighbours[ndx].z = INVALIDID;

  __global uint3 *nb = &(mesh->neighbours[nb1]);

  if (mesh->neighbours[nb1].x == oldt) {
    nb->x = ndx;
  }
  else {
    if (mesh->neighbours[nb1].y == oldt)
      nb->y = ndx;
    else {
      nb->z = ndx;
    }
  }

  return ndx;
}

int adjacent(uint3 elem1, uint3 elem2) {

  int sc = 0;
  if (elem1.x == elem2.x || elem1.x == elem2.y || elem1.x == elem2.z)
    sc++;

  if (elem1.y == elem2.x || elem1.y == elem2.y || elem1.y == elem2.z)
    sc++;

  if (!IS_SEGMENT(elem1) && (elem1.z == elem2.x || elem1.z == elem2.y || elem1.z == elem2.z))
    sc++;

  return sc == 2;
}

void addneighbour(__global uint3 *neigh, uint elem) {

  if (neigh->x == elem || neigh->y == elem || neigh->z == elem) return;

  if (neigh->x == INVALIDID) { neigh->x = elem; return; }
  if (neigh->y == INVALIDID) { neigh->y = elem; return; }
  if (neigh->z == INVALIDID) { neigh->z = elem; return; }
}



void setup_neighbours(__global Mesh *mesh, uint start, uint end) {

  // Relies on all neighbours being in start--end
  for (uint i = start; i < end; i++) {
    __global uint3 *neigh = &(mesh->neighbours[i]);

    for (uint j = i+1; j < end; j++) {
      __global uint3 *neigh2 = &(mesh->neighbours[j]);

      if (adjacent((mesh->elements[i]), (mesh->elements[j]))) {
        addneighbour(neigh, j);
        addneighbour(neigh2, i);
      }
    }
  }
}
