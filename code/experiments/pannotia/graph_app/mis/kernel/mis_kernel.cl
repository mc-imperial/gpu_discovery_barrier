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

// This application is the original Pannotia mis application with the
// addition of a "mega kernel" which allows all computation to be
// done on the GPU

// For the discovery protocol and inter-workgroup barrier
#include "discovery.cl"

#define BIGNUM 99999999

/**
 * init kernel
 * @param s_array   set array
 * @param c_array   status array
 * @param cu_array  status update array
 * @param num_nodes number of vertices
 * @param num_edges number of edges
 */
__kernel void init(__global int *s_array,
                   __global int *c_array,
                   __global int *cu_array,
                   int num_nodes,
                   int num_edges) {

  // Get my workitem id
  int tid = get_global_id(0);
  if (tid < num_nodes) {

    // Set the status array: not processed
    c_array[tid]  = -1;
    cu_array[tid] = -1;
    s_array[tid]  =  0;
  }
}

/**
 * mis1 kernel
 * @param row          csr pointer array
 * @param col          csr column index array
 * @param node_value   node value array
 * @param s_array      set array
 * @param c_array node status array
 * @param min_array    node value array
 * @param stop node    value array
 * @param num_nodes    number of vertices
 * @param num_edges    number of edges
 */
__kernel void mis1(  __global int *row,
                     __global int *col,
                     __global float *node_value,
                     __global int *s_array,
                     __global int *c_array,
                     __global float *min_array,
                     __global int *stop,
                     int num_nodes,
                     int num_edges) {

  // Get my workitem id
  int tid = get_global_id(0);
  if (tid < num_nodes) {

    // If the vertex is not processed
    if (c_array[tid] == -1) {
      *stop = 1;

      // Get the start and end pointers
      int start = row[tid];
      int end;
      if (tid + 1 < num_nodes)
        end = row[tid + 1] ;
      else
        end = num_edges;

      // Navigate the neighbor list and find the min
      float min = BIGNUM;
      for (int edge = start; edge < end; edge++) {
        if (c_array[col[edge]] == -1) {
          if (node_value[col[edge]] < min)
            min = node_value[col[edge]];
        }
      }
      min_array[tid] = min;
    }
  }
}

/**
 * mis2 kernel
 * @param row          csr pointer array
 * @param col          csr column index array
 * @param node_value   node value array
 * @param s_array      set array
 * @param c_array      status array
 * @param cu_array     status update array
 * @param min_array    node value array
 * @param num_nodes    number of vertices
 * @param num_edges    number of edges
 */
__kernel void  mis2(  __global int *row,
                      __global int *col,
                      __global float *node_value,
                      __global int *s_array,
                      __global int *c_array,
                      __global int *cu_array,
                      __global float *min_array,
                      int num_nodes,
                      int num_edges) {
  // Get my workitem id
  int tid = get_global_id(0);
  if (tid < num_nodes) {
    if (node_value[tid] < min_array[tid]  && c_array[tid] == -1) {

      // -1 : not processed
      // -2 : inactive.
      //  2 : independent set - put the item into the independent set
      s_array[tid] = 2;

      // Get the start and end pointers
      int start = row[tid];
      int end;

      if (tid + 1 < num_nodes)
        end = row[tid + 1] ;
      else
        end = num_edges;

      // Set the status to inactive
      c_array[tid] = -2;

      // Mark all the neighnors inactive
      for (int edge = start; edge < end; edge++) {
        if (c_array[col[edge]] == -1)
          // Use status update array to avoid race
          cu_array[col[edge]] = -2;
      }

    }
  }
}

/**
 * mis3 kernel
 * @param cu_array     status update array
 * @param  c_array     status array
 * @param num_nodes    number of vertices
 */
__kernel void  mis3(  __global int *cu_array,
                      __global int *c_array,
                      int num_nodes) {

  // Get my workitem id
  int tid = get_global_id(0);

  // Set the status array
  if (tid < num_nodes && cu_array[tid] == -2)
    c_array[tid] = cu_array[tid];
}

// mega_kernel: combines the mis1, mis2, and mis3 kernels using an
// inter-workgroup barrier and the discovery protocol
__kernel void mega_kernel( __global int *row,
                           __global int *col,
                           __global float *node_value,
                           __global int *s_array,
                           __global int *c_array,
                           __global int *cu_array,
                           __global float *min_array,
                           __global int *stop,
                           int num_nodes,
                           int num_edges,
                           __global discovery_kernel_ctx *gl_ctx
                           ) {

  DISCOVERY_PROTOCOL(gl_ctx);

  // Get participating global id and the stride
  int tid_start = p_get_global_id(gl_ctx, &local_ctx);
  int stride = p_get_global_size(gl_ctx, &local_ctx);
  int local_stop;

  while(1) {

    // Original application --- mis1 --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int tid = tid_start; tid < num_nodes; tid += stride) {

      // If the vertex is not processed
      if (c_array[tid] == -1) {
        *stop = 1;

        // Get the start and end pointers
        int start = row[tid];
        int end;
        if (tid + 1 < num_nodes)
          end = row[tid + 1] ;
        else
          end = num_edges;

        // Navigate the neighbor list and find the min
        float min = BIGNUM;
        for(int edge = start; edge < end; edge++) {
          if (c_array[col[edge]] == -1) {
            if (node_value[col[edge]] < min)
              min = node_value[col[edge]];
          }
        }
        min_array[tid] = min;
      }
    }

    // Original application --- mis1 --- end

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);
    local_stop = *stop;

    // Original application --- mis2 --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int tid = tid_start; tid < num_nodes; tid += stride) {

      if (node_value[tid] < min_array[tid]  && c_array[tid] == -1) {

        // -1 : not processed
        // -2 : inactive
        //  2 : independent set put the item into the independent set
        s_array[tid] = 2;

        // Get the start and end pointers
        int start = row[tid];
        int end;

        if (tid + 1 < num_nodes)
          end = row[tid + 1] ;
        else
          end = num_edges;

        // Set the status to inactive
        c_array[tid] = -2;

        // Mark all the neighnors inactive
        for(int edge = start; edge < end; edge++) {
          if (c_array[col[edge]] == -1) {

            // Use status update array to avoid race
            cu_array[col[edge]] = -2;
          }
        }
      }
    }

    // Original application --- mis2 --- end

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);


    if (local_stop == 0) {
      break;
    }
    *stop = 0;

    // Original application --- mis3 --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int tid = tid_start; tid < num_nodes; tid += stride) {
      if (cu_array[tid] == -2) {
        c_array[tid] = cu_array[tid];
      }
    }

    // Original application --- mis3 --- end

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);
  }
}
//