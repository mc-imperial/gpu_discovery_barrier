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

// The GPU-Lonestar dmr application ported to OpenCL (device part).
// Port by Tyler Sorensen (2016)

// This application uses the portable discovery protocol which
// requires no knowledge about the target GPU occupancy

#include "dmr_common.cl"

__kernel void refine(__global Mesh *mesh,
                     __global uint *nnodes,
                     __global uint *nelements,
                     __global CL_Worklist2 *wl,
                     __global CL_Worklist2 *owl,
                     __global discovery_kernel_ctx *gl_ctx
                     ) {

  DISCOVERY_PROTOCOL(gl_ctx);

  int id = p_get_global_id(gl_ctx, &local_ctx);
  int threads = p_get_global_size(gl_ctx, &local_ctx);
  int ele, eleit, haselem;

  uint cavity[CAVLEN], nc = 0;
  uint boundary[BCLEN], bc = 0;
  uint ulimit = ((*(wl->dindex) + threads - 1) / threads) * threads;
  int repush = 0;

  const int perthread = ulimit / threads;
  int stage = 0;
  int x = 0;

  for (eleit = id * perthread; eleit < (id * perthread + perthread) && eleit < ulimit; eleit++, x++) {

    FORD cx, cy;
    haselem = wl_pop_id(wl, eleit, &ele);
    nc = 0;
    bc = 0;
    stage = 0;
    repush = 0;

    if (haselem && ele < mesh->nelements && mesh->isbad[ele] && !mesh->isdel[ele]) {
      cavity[nc++] = ele;
      uint oldcav;

      do {
        oldcav = cavity[0];
        cavity[0] = opposite(mesh, ele);
      } while (cavity[0] != oldcav);

      if (!build_cavity(mesh, cavity, &nc, CAVLEN, boundary, &bc, &cx, &cy)) {
        build_cavity(mesh, cavity, &nc, CAVLEN, boundary, &bc, &cx, &cy);
      }

      // Try to claim ownership
      for (int i = 0; i < nc; i++) {
        mesh->owners[cavity[i]] = id;
      }

      for (int i = 0; i < bc; i+=4) {
        mesh->owners[boundary[i + 2]] =  id;
      }

      stage = 1;
    }

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);

    if (stage == 1) {

      // Check for conflicts
      for (int i = 0; i < nc; i++) {
        if (mesh->owners[cavity[i]] != id)
          atomic_min(&mesh->owners[cavity[i]], id);
      }

      for (int i = 0; i < bc; i+=4) {
        if (mesh->owners[boundary[i + 2]] != id) {
          atomic_min(&mesh->owners[boundary[i + 2]], id);
        }
      }

      stage = 2;
    }

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);

    int nodes_added = 0;
    int elems_added = 0;

    if (stage == 2) {
      int i;

      for (i = 0; i < nc; i++) {

        if (mesh->owners[cavity[i]] != id) {
          repush = 1;
          break;
        }
      }

      if (!repush)
        for (i = 0; i < bc; i+=4) {
          if (mesh->owners[boundary[i + 2]] != id) {
            repush = 1;
            break;
          }
        }

      if (!repush) {
        stage = 3;
        nodes_added = 1;
        elems_added = (bc >> 2) + (IS_SEGMENT(mesh->elements[cavity[0]]) ? 2 : 0);
      }
    }

    if (stage == 3) {

      uint cnode = add_node(mesh, cx, cy, atomic_add(nnodes, 1));
      uint cseg1 = 0, cseg2 = 0;

      uint nelements_added = elems_added;
      uint oldelements = atomic_add(nelements, nelements_added);

      uint newelemndx = oldelements;
      if (IS_SEGMENT(mesh->elements[cavity[0]])) {
        cseg1 = add_segment(mesh, mesh->elements[cavity[0]].x, cnode, newelemndx++);
        cseg2 = add_segment(mesh, cnode, mesh->elements[cavity[0]].y, newelemndx++);
      }

      for (int i = 0; i < bc; i+=4) {
        add_triangle(mesh, boundary[i], boundary[i+1], cnode, boundary[i+2], boundary[i+3], newelemndx++);
      }

      setup_neighbours(mesh, oldelements, newelemndx);

      repush = 1;
      for (int i = 0; i < nc; i++) {
        mesh->isdel[cavity[i]] = 1;

        // If the resulting cavity does not contain the original triangle
        // (because of the opposite() routine, add it back.
        if(cavity[i] == ele) {
          repush = 0;
        }
      }
    }

    if (repush) {
      wl_push(owl, ele);
    }

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);
  }
  mesh->nnodes = *nnodes;
  mesh->nelements = *nelements;
}
//
