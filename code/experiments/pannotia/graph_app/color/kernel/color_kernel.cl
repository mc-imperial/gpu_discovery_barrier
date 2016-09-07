/************************************************************************************\
 *                                                                                  *
 * Copyright  2014 Advanced Micro Devices, Inc.                                     *
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
 * @brief   color kernel 1
 * @param   row         CSR pointer array
 * @param   col         CSR column array
 * @param   node_value  Vertex value array
 * @param   color_array Color value array
 * @param   stop        Termination variable
 * @param   max_d       Max array
 * @param   color       Current color label
 * @param   num_nodes   Number of vertices
 * @param   num_edges   Number of edges
 */
__kernel  void color( __global int   *row,
                      __global int   *col,
                      __global float *node_value,
                      __global int   *color_array,
                      __global int   *stop,
                      __global float *max_d,
                      const  int color,
                      const  int num_nodes,
                      const  int num_edges) {

  // Get my thread workitem id
  int tid = get_global_id(0);

  if (tid < num_nodes) {

    // If the vertex is still not colored
    if (color_array[tid] == -1) {

      // Get the start and end pointer of the neighbor list
      int start = row[tid];
      int end;
      if (tid + 1 < num_nodes)
        end = row[tid + 1];
      else
        end = num_edges;

      float maximum = -1;

      // Navigate the neighbor list
      for (int edge = start; edge < end; edge++) {

        // Determine if the vertex value is the maximum in the neighborhood
        if (color_array[col[edge]] == -1 && start != end - 1) {
          *stop = 1;
          if (node_value[col[edge]] > maximum)
            maximum = node_value[col[edge]];
        }
      }

      // Assign maximum the max array
      max_d[tid] = maximum;
    }
  }
}

/**
 * @brief   color kernel 2
 * @param   node_value  Vertex value array
 * @param   color_array Color value array
 * @param   max_d       Max array
 * @param   color       Current color label
 * @param   num_nodes   Number of vertices
 * @param   num_edges   Number of edges
 */
__kernel  void color2( __global float *node_value,
                       __global int   *color_array,
                       __global float *max_d,
                       const int color,
                       const int num_nodes,
                       const int num_edges){

  // Get my workitem id
  int tid = get_global_id(0);

  if (tid < num_nodes) {

    // If the vertex is still not colored
    if (color_array[tid] == -1) {
      if (node_value[tid] > max_d[tid])

        //Assign a color
        color_array[tid] = color;
    }
  }
}


// mega_kernel: combines the color1 and color2 kernels using an
// inter-workgroup barrier and the discovery protocol
__kernel void mega_kernel( __global int   *row,                        //0
                           __global int   *col,                        //1
                           __global float *node_value,                 //2
                           __global int   *color_array,                //3
                           __global int   *stop1,                      //4
                           __global int   *stop2,                      //5
                           __global float *max_d,                      //6
                           const  int num_nodes,                       //7
                           const  int num_edges,                       //8
                           __global discovery_kernel_ctx *gl_ctx) {    //9

  DISCOVERY_PROTOCOL(gl_ctx);
  __global int * write_stop = stop1;
  __global int * read_stop = stop2;
  __global int * swap;

  // Get global participating group id and the stride
  int tid = p_get_global_id(gl_ctx, &local_ctx);
  int stride = p_get_global_size(gl_ctx, &local_ctx);
  int graph_color = 1;

  while (1) {

    // Original application --- color --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_nodes; i+=stride) {

      // If the vertex is still not colored
      if (color_array[i] == -1) {

        // Get the start and end pointer of the neighbor list
        int start = row[i];
        int end;
        if (i + 1 < num_nodes)
          end = row[i + 1];
        else
          end = num_edges;

        float maximum = -1;

        // Navigate the neighbor list
        for (int edge = start; edge < end; edge++) {

          // Determine if the vertex value is the maximum in the neighborhood
          if (color_array[col[edge]] == -1 && start != end - 1) {
            *write_stop = 1;
            if (node_value[col[edge]] > maximum)
              maximum = node_value[col[edge]];
          }
        }
        // Assign maximum the max array
        max_d[i] = maximum;
      }
    }

    // Two terminating variables allow us to only use 1
    // inter-workgroup barrier and still avoid a data-race
    swap = read_stop;
    read_stop = write_stop;
    write_stop = swap;

    // Original application --- color --- end

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);

    // Original application --- color2 --- start

    // The original kernels used an 'if' here. We need a 'for' loop
    for (int i = tid; i < num_nodes; i+=stride) {

      // If the vertex is still not colored
      if (color_array[i] == -1) {
        if (node_value[i] > max_d[i])

          // Assign a color
          color_array[i] = graph_color;
      }
    }

    if (*read_stop == 0) {
      break;
    }

    graph_color = graph_color + 1;
    *write_stop = 0;

    // Original application --- color2 --- end

    // Inter-workgroup barrier
    discovery_barrier(gl_ctx, &local_ctx);
  }
}
//