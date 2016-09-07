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

// A file for parsing and dealing with mesh files.
// A platform portable (Windows and Linux) implementation
// by Tyler Sorensen (2016)

#pragma once

#include <string>
#include "shmesh.h"
#include <fstream>
#include <limits>
#include <iostream>

// Needed for windows
#undef max
#undef min

void next_line(std::ifstream& scanner) {
  scanner.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

void readNodes(std::string filename, ShMesh &mesh, int maxfactor = 2) {
  size_t index;
  FORD x, y;
  bool firstindex = true;

  std::ifstream scanner(filename.append(".node").c_str());
  scanner >> mesh.nnodes;

  mesh.maxnnodes = (maxfactor /  MAX_NNODES_TO_NELEMENTS) * mesh.nnodes;
  printf("memory for nodes: %lu MB\n", mesh.maxnnodes * sizeof(FORD) * 2 / 1048576);
  mesh.nodex = (FORD *) malloc(sizeof(FORD) * mesh.maxnnodes);
  mesh.nodey = (FORD *) malloc(sizeof(FORD) * mesh.maxnnodes);

  FORD *nodex = mesh.nodex;
  FORD *nodey = mesh.nodey;

  for (size_t i = 0; i < mesh.nnodes; i++) {
    next_line(scanner);
    scanner >> index >> x >> y;
    if(firstindex) { assert(index == 0); firstindex = false;}

    nodex[index] = x;
    nodey[index] = y;
  }
}

void readTriangles(std::string basename, ShMesh &mesh, int maxfactor = 2) {
  unsigned ntriangles, nsegments;
  unsigned i, index, n1, n2, n3;
  bool firstindex = true;
  std::string filename;

  filename = basename;
  std::ifstream scanner(filename.append(".ele").c_str());
  scanner >> ntriangles;

  filename = basename;
  std::ifstream scannerperimeter(filename.append(".poly").c_str());
  scannerperimeter >> nsegments; // First line is number of nodes
  assert(nsegments == 0);        // Standard triangle format, nodes == 0
  next_line(scannerperimeter);
  scannerperimeter >> nsegments; // Number of segments

  mesh.ntriangles = ntriangles;
  mesh.nsegments = nsegments;
  mesh.nelements = ntriangles + nsegments;
  mesh.maxnelements = maxfactor * mesh.nelements;

  printf("memory for elements: %lu MB\n", mesh.maxnelements * (sizeof(cl_uint3) * 2 + sizeof(int) * 2) / 1048576);
  mesh.elements = (cl_uint3*) malloc(sizeof(cl_uint3) * mesh.maxnelements);
  mesh.isdel = (int*) malloc(sizeof(int) * mesh.maxnelements);
  mesh.isbad = (int*) malloc(sizeof(int) * mesh.maxnelements);
  mesh.neighbours = (cl_uint3*) malloc(sizeof(cl_uint3) * mesh.maxnelements);

  cl_uint3 *elements = mesh.elements;
  int *isdel = mesh.isdel, *isbad = mesh.isbad;
  for (i = 0; i < ntriangles; i++) {
    next_line(scanner);
    scanner >> index >> n1 >> n2 >> n3;
    if(firstindex) { assert(index == 0); firstindex = false;}

    elements[index].x = n1;
    elements[index].y = n2;
    elements[index].z = n3;
    isdel[index] = isbad[index] = 0;
  }

  firstindex = true;
  for (i = 0; i < nsegments; i++) {
    next_line(scannerperimeter);
    scannerperimeter >> index >> n1 >> n2;

    if(firstindex) {
      assert(index == 0);
      firstindex = false;
    }

    elements[index + ntriangles].x = n1;
    elements[index + ntriangles].y = n2;
    elements[index + ntriangles].z = INVALIDID;
    isdel[index] = isbad[index] = false;
  }
}

void write_mesh(std::string infile, ShMesh &mesh) {

  FORD *nodex, *nodey;

  nodex = mesh.nodex;
  nodey = mesh.nodey;

  unsigned slash = infile.rfind("/");
  std::cout << "  -- " << infile.substr(slash + 1) + ".out.node (" << mesh.nnodes << " nodes)" << std::endl;
  std::ofstream outfilenode((infile.substr(slash + 1) + ".out.node").c_str());
  outfilenode.precision(17);
  outfilenode << mesh.nnodes << " 2 0 0\n";
  for (size_t ii = 0; ii < mesh.nnodes; ++ii) {
    outfilenode << ii << " " << nodex[ii] << " " << nodey[ii] << "\n";
  }
  outfilenode.close();

  cl_uint3 *elements = mesh.elements;
  int *isdel = mesh.isdel;

  unsigned ntriangles2 = mesh.nelements;
  unsigned segmentcnt = 0;
  for (size_t ii = 0; ii < mesh.nelements; ++ii) {

    if(IS_SEGMENT(elements[ii]) || isdel[ii])
      ntriangles2--;

    if(IS_SEGMENT(elements[ii]) && !isdel[ii])
      segmentcnt++;
  }

  std::cout << "  -- " << infile.substr(slash + 1) + ".out.ele (" << ntriangles2 << " triangles)" << std::endl;
  std::ofstream outfileele((infile.substr(slash + 1) + ".out.ele").c_str());

  outfileele << ntriangles2 << " 3 0\n";
  size_t kk = 0;
  for (size_t ii = 0; ii < mesh.nelements; ++ii) {

    if(!IS_SEGMENT(elements[ii]) && !isdel[ii])
      outfileele << kk++ << " " << elements[ii].x << " " << elements[ii].y << " " << elements[ii].z << "\n";
  }
  outfileele.close();

  std::cout << "  -- " << infile.substr(slash + 1) + ".out.poly (" << segmentcnt << " segments)" <<std::endl;
  std::ofstream outfilepoly((infile.substr(slash + 1) + ".out.poly").c_str());
  outfilepoly << "0 2 0 1\n";
  outfilepoly << segmentcnt << " 0\n";
  kk = 0;
  for (size_t ii = 0; ii < mesh.nelements; ++ii) {

    if(IS_SEGMENT(elements[ii]) && !isdel[ii])
      outfilepoly << kk++ << " " << elements[ii].x << " " << elements[ii].y << "\n";
  }
  outfilepoly << "0\n";
  outfilepoly.close();

  std::cout << (ntriangles2 + segmentcnt) << " active elements of " << mesh.nelements << " total elements (" << mesh.nelements / (ntriangles2 + segmentcnt) << "x) " << std::endl;
  std::cout << 1.0 * mesh.maxnelements / mesh.nelements << " ratio of used to free elements." << std::endl;
}
