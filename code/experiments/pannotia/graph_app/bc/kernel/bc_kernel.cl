/************************************************************************************\
 *                                                                                  *
 * Copyright © 2014 Advanced Micro Devices, Inc.                                    *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR") (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/

// File modified by Tyler Sorensen (2016) to run the experiments for
// the OOPSLA'16 paper "Portable Inter-Workgroup Barrier
// Synchronisation for GPUs"

// This application is the original Pannotia BC application with the
// addition of a "mega kernel" which allows all computation to be
// done on the GPU

// For the discovery protocol and inter-workgroup barrier
#include "discovery.cl"

/**
 * @brief   atomic add float
 * @param   address   Address
 * @param   val       value to add
 */
inline float atomic_add_float(__global float* const address,
                              const float value) {

  uint oldval, newval, readback;

  *(float*)&oldval = *address;
  *(float*)&newval = (*(float*)&oldval + value);
  while ((readback = atomic_cmpxchg((__global uint *) address, oldval, newval)) != oldval) {
    oldval = readback;
    *(float*) &newval = (*(float*) &oldval + value);
  }
  return *(float*) &oldval;
}

/**
 * @brief   array set 1D
 * @param   s           Source vertex
 * @param   dist_array  Distance array
 * @param   sigma       Sigma array
 * @param   rho         Rho array
 * @param   num_nodes Termination variable
 */
__kernel void clean_1d_array(const int source,
                             __global int *dist_array,
                             __global float *sigma,
                             __global float *rho,
                             const int num_nodes) {

  int tid = get_global_id(0);

  if (tid < num_nodes) {
    sigma[tid] = 0;

    // If source vertex rho = 1, dist = 0
    if (tid == source) {
      rho[tid] = 1;
      dist_array[tid] = 0;

    } else { // If other vertices rho = 0, dist = -1
      rho[tid] = 0;
      dist_array[tid] = -1;
    }
  }
}

/**
 * @brief   array set 2D
 * @param   p           Dependency array
 * @param   num_nodes   Number of vertices
 */
__kernel void clean_2d_array( __global int *p,
                              const int num_nodes) {

  int tid = get_global_id(0);

  if (tid < num_nodes * num_nodes)
    p[tid] = 0;
}

/**
 * @brief   clean BC
 * @param   bc_d        Betweeness Centrality array
 * @param   num_nodes   Number of vertices
 */
__kernel void clean_bc( __global float *bc_d,
                        const  int num_nodes) {

  int tid = get_global_id(0);
  if (tid < num_nodes)
    bc_d[tid] = 0;
}

// Only include the non inter-workgroup barrier code if we need it.
// Having to much code leads to crashes for some compilers (!!)
#ifndef GB_VAR


/**
 * @brief   Breadth-first traversal
 * @param   row       CSR pointer array
 * @param   col       CSR column  array
 * @param   d         Distance array
 * @param   rho       Rho array
 * @param   p         Dependency array
 * @param   cont      Termination variable
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 * @param   dist      Current traversal layer
 */

__kernel void bfs_kernel(__global int   *row,
                          __global int   *col,
                          __global int   *d,
                          __global float *rho,
                          __global int   *p,
                          __global int   *cont,
                          const    int    num_nodes,
                          const    int    num_edges,
                          const    int    dist) {

  int tid = get_global_id(0);

  // Navigate the current layer
  if (tid < num_nodes && d[tid] == dist) {

    // Get the starting and ending pointers
    // of the neighbor list
    int start = row[tid];
    int end;
    if (tid + 1 < num_nodes)
      end = row[tid + 1] ;
    else
      end = num_edges;

    // Navigate through the neighbor list
    for (int edge = start; edge < end; edge++) {
      int w = col[edge];
      if (d[w] < 0) {
        *cont = 1;

        // Traverse another layer
        d[w] = dist + 1;
      }

      // Transfer the rho value to the neighbor
      if (d[w] == (dist + 1)) {
        atomic_add_float(&rho[w], rho[tid]);
      }
    }
  }
}

/**
 * @brief   Back traversal
 * @param   row       CSR pointer array
 * @param   col       CSR column  array
 * @param   d         Distance array
 * @param   rho       Rho array
 * @param   sigma     Sigma array
 * @param   p         Dependency array
 * @param   cont      Termination variable
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 * @param   dist      Current traversal layer
 * @param   s         Source vertex
 * @param   bc        Betweeness Centrality array
 */

__kernel void backtrack_kernel(__global  int *row,
                               __global  int *col,
                               __global  int *d,
                               __global  float *rho,
                               __global  float *sigma,
                               __global  int *p,
                               const     int num_nodes,
                               const     int num_edges,
                               const     int dist,
                               const     int s,
                               __global  float* bc){


  int tid = get_global_id(0);

  // Navigate the current layer
  if (tid < num_nodes && d[tid] == dist-1) {

    int start = row[tid];
    int end;
    if (tid + 1 < num_nodes)
      end = row[tid + 1] ;
    else
      end = num_edges;

    // Get the starting and ending pointers
    // of the neighbor list in the reverse graph
    for (int edge = start; edge < end; edge++) {
      int w = col[edge];

      // Update the sigma value traversing back
      if (d[w] == dist - 2)
        atomic_add_float(&sigma[w], rho[w]/rho[tid] * (1 + sigma[tid]));
    }

    // Update the BC value
    if (tid!=s)
      bc[tid] = bc[tid] + sigma[tid];
  }
}

/**
 * @brief   back_sum_kernel (not used)
 * @param   s         Source vertex
 * @param   dist      Current traversal layer
 * @param   d         Distance array
 * @param   sigma     Sigma array
 * @param   bc        Betweeness Centrality array
 * @param   num_nodes Termination variable
 * @param   num_edges Termination variable
 */
__kernel void back_sum_kernel(   const    int s,
                                 const    int dist,
                                 __global int *d,
                                 __global float *sigma,
                                 __global float *bc,
                                 const    int num_nodes){

  int tid = get_global_id(0);

  if (tid < num_nodes) {

    // If it is not the source
    if (s != tid && d[tid] == dist - 1) {
      bc[tid] = bc[tid] + sigma[tid];
    }
  }
}


#else // defined(GB_VAR)

// Here we have the mega-kernel code. That is, the kernels
// above combined and the host side loop combined on the GPU
int mega_bfs_kernel_func( __global int   *row,
                          __global int   *col,
                          __global int   *d,
                          __global float *rho,
                          __global int   *p,
                          __global int   *stop1,
                          __global int   *stop2,
                          __global int   *stop3,
                          const    int    num_nodes,
                          const    int    num_edges,
                          __global int   *global_dist,
                          __global discovery_kernel_ctx *gl_ctx,
                          __local  discovery_local_ctx  *local_ctx) {

  __global int * write_stop = stop1;
  __global int * read_stop  = stop2;
  __global int * buff_stop  = stop3;
  __global int * swap;

  // Get participating global id and the stride
  int tid        = p_get_global_id(gl_ctx, local_ctx);
  int stride     = p_get_global_size(gl_ctx, local_ctx);
  int local_dist = 0;

  while(1) {

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_nodes; i+=stride) {
      if (d[i] == local_dist) {

        // Get the starting and ending pointers
        // of the neighbor list
        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1] ;
        else
          end = num_edges;

        // Navigate through the neighbor list
        for (int edge = start; edge < end; edge++) {
          int w = col[edge];
          if (d[w] < 0) {
            *write_stop = 1;

            // Traverse another layer
            d[w] = local_dist + 1;
          }

          // Transfer the rho value to the neighbor
          if (d[w] == (local_dist + 1)) {
            atomic_add_float(&rho[w], rho[i]);
          }
        }
      }
    }

    swap       = read_stop;
    read_stop  = write_stop;
    write_stop = buff_stop;
    buff_stop  = swap;
    local_dist = local_dist + 1;

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, local_ctx);

    // Trick for updating the termination variables using the
    // 'buff_stop' variable without extra barriers and avoiding
    // data-races.
    if (*read_stop == 0) {
      break;
    }
    *buff_stop = 0;
  }

  return local_dist;
}

void mega_backtrack_kernel_func(__global int   *row,
                                __global int   *col,
                                __global int   *d,
                                __global float *rho,
                                __global float *sigma,
                                __global int   *p,
                                const    int    num_nodes,
                                const    int    num_edges,
                                const    int    dist,
                                const    int    s,
                                __global float *bc,
                                __global discovery_kernel_ctx *gl_ctx,
                                __local  discovery_local_ctx  *local_ctx) {


  // Get global participating id and stride
  int tid        = p_get_global_id(gl_ctx, local_ctx);
  int stride     = p_get_global_size(gl_ctx, local_ctx);
  int local_dist = dist;

  while (local_dist > 0) {

    for (int i = tid; i < num_nodes; i += stride) {
      if (d[i] == local_dist - 1) {

        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1] ;
        else
          end = num_edges;

        // Get the starting and ending pointers
        // of the neighbor list in the reverse graph
        for (int edge = start; edge < end; edge++) {
          int w = col[edge];

          // Update the sigma value traversing back
          if (d[w] == local_dist - 2)
            atomic_add_float(&sigma[w], rho[w]/rho[i] * (1 + sigma[i]));
        }

        // Update the BC value

        // Tyler: This looks like there might be a data-race here, but
        // the original authors assured me that there isn't.
        if (i!=s)
          bc[i] = bc[i] + sigma[i];
      }
    }
    local_dist = local_dist - 1;

    // Inter workgroup barrier
    discovery_barrier(gl_ctx, local_ctx);
  }
}


__kernel void mega_bc_kernel(__global int   *row,                      // 0
                             __global int   *col,                      // 1
                             __global int   *row_trans,                // 2
                             __global int   *col_trans,                // 3
                             __global int   *dist,                     // 4
                             __global float *rho,                      // 5
                             __global float *sigma,                    // 6
                             __global int   *p,                        // 7
                             __global int   *stop1,                    // 8
                             __global int   *stop2,                    // 9
                             __global int   *stop3,                    // 10
                             __global int   *global_dist,              // 11
                             __global float *bc,                       // 12
                             const    int    num_nodes,                // 13
                             const    int    num_edges,                // 14
                             __global discovery_kernel_ctx *gl_ctx  // 15
                             ) {
  // Discovery protocol
  DISCOVERY_PROTOCOL(gl_ctx);

  // Original application --- clean_1d_array --- start

  int tid    = p_get_global_id(gl_ctx, &local_ctx);
  int stride = p_get_global_size(gl_ctx, &local_ctx) * get_local_size(0);

  for (int s = 0; s < num_nodes; s++) {

    for (int i = tid; i < num_nodes; i+=stride) {
      sigma[i] = 0;

      // If source vertex rho = 1, dist = 0
      if (i == s) {
        rho[i]  = 1;
        dist[i] = 0;

      } else { // If other vertices rho = 0, dist = -1
        rho[i]  = 0;
        dist[i] = -1;
      }
    }

    // Original application --- clean 1d_array --- end

    // No barrier required here because the two kernels
    // access disjoint memory.

    // Original application --- clean 2d_array --- start

    for (int i = tid; i < num_nodes * num_nodes; i+=stride) {
      p[i] = 0;
    }

    // Original application --- clean 2d_array --- end

    // Inter workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);

    // Original application --- bfs_kernel --- start

    int local_dist = mega_bfs_kernel_func(row, col, dist,
                                          rho, p, stop1, stop2,
                                          stop3, num_nodes,
                                          num_edges, global_dist,
                                          gl_ctx, &local_ctx);

    // Original application --- bfs_kernel --- end

    // Inter workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);

    // Original application --- backtrack_kernel --- start

    mega_backtrack_kernel_func(row_trans, col_trans, dist,
                               rho, sigma, p, num_nodes,
                               num_edges, local_dist, s, bc,
                               gl_ctx, &local_ctx);

    // Original application --- backtrack_kernel --- end
  }

  // Inter workgroup barrier
  discovery_barrier(gl_ctx, &local_ctx);
}

#endif
//