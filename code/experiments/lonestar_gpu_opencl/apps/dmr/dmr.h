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

// These are functions common to both the portable and non-portable
// version

// Clean up OpenCL utilities
void clean_opencl() {
  clReleaseCommandQueue(queue);
  clReleaseProgram(prog);
  clReleaseContext(context);
}

// Initialise OpenCL utilities
void init_opencl() {
  int err;
  device = create_device();
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERR(err);
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERR(err);
  char opts[500];
  get_compile_opts(opts);
  prog = build_program(context, device, CL_FILE, opts);
}

void addneighbour_cpu(cl_uint3 &neigh, uint elem) {
  if (neigh.x == elem || neigh.y == elem || neigh.z == elem)
    return;

  assert(neigh.x == INVALIDID || neigh.y == INVALIDID || neigh.z == INVALIDID);

  if (neigh.x == INVALIDID) { neigh.x = elem; return; }
  if (neigh.y == INVALIDID) { neigh.y = elem; return; }
  if (neigh.z == INVALIDID) { neigh.z = elem; return; }
}


void find_neighbours_cpu(ShMesh &mesh) {

  std::map<std::pair<int, int>, int> edge_map;
  uint nodes1[3];

  cl_uint3 *elements = mesh.elements;
  cl_uint3 *neighbours = mesh.neighbours;
  int ele;

  for (ele = 0; ele < mesh.nelements; ele++) {
    cl_uint3 *neigh = &neighbours[ele];

    neigh->x = INVALIDID;
    neigh->y = INVALIDID;
    neigh->z = INVALIDID;

    nodes1[0] = elements[ele].x;
    nodes1[1] = elements[ele].y;
    nodes1[2] = elements[ele].z;

    if(nodes1[0] > nodes1[1]) std::swap(nodes1[0], nodes1[1]);
    if(nodes1[1] > nodes1[2]) std::swap(nodes1[1], nodes1[2]);
    if(nodes1[0] > nodes1[1]) std::swap(nodes1[0], nodes1[1]);

    assert(nodes1[0] <= nodes1[1] && nodes1[1] <= nodes1[2]);

    // The Windows compiler infers types for std::make_pair, and
    // complains if types are explicitly given. This is a work
    // around so that this function can work on Windows and Linux
#ifdef WIN32
    std::pair<int, int> edges[3];
    edges[0] = std::make_pair(nodes1[0], nodes1[1]);
    edges[1] = std::make_pair(nodes1[1], nodes1[2]);
    edges[2] = std::make_pair(nodes1[0], nodes1[2]);
#else
    std::pair<int, int> edges[3];
    edges[0] = std::make_pair<int, int>(nodes1[0], nodes1[1]);
    edges[1] = std::make_pair<int, int>(nodes1[1], nodes1[2]);
    edges[2] = std::make_pair<int, int>(nodes1[0], nodes1[2]);
#endif

    int maxn = IS_SEGMENT(elements[ele]) ? 1 : 3;

    for (int i = 0; i < maxn; i++) {

      if (edge_map.find(edges[i]) == edge_map.end())
        edge_map[edges[i]] = ele;
      else {
        int node = edge_map[edges[i]];
        addneighbour_cpu(neighbours[node], ele);
        addneighbour_cpu(neighbours[ele], node);
        edge_map.erase(edges[i]);
      }
    }
  }
}

void free_mesh(ShMesh *mesh) {
  free(mesh->nodex);
  free(mesh->nodey);
  free(mesh->elements);
  free(mesh->neighbours);
  free(mesh->isdel);
  free(mesh->isbad);
  free(mesh->owners);
}

void read_mesh(const char *basefile, ShMesh &mesh, int maxfactor) {
  readNodes(basefile, mesh, maxfactor);
  readTriangles(basefile, mesh, maxfactor);

  assert(mesh.maxnelements > 0);
  printf("memory for owners: %lu MB\n", mesh.maxnelements * sizeof(int) / 1048576);
  mesh.owners = (int *) malloc(sizeof(int) * mesh.maxnelements);

  // See refine() for actual allocation
  printf("memory for worklists: %lu MB\n", 2 * mesh.nelements * sizeof(int) / 1048576);

  printf("%s: %d nodes, %d triangles, %d segments read\n", basefile, mesh.nnodes, mesh.ntriangles, mesh.nsegments);
  assert(mesh.nnodes > 0);
  assert(mesh.ntriangles > 0);
  assert(mesh.nsegments > 0);
  assert(mesh.nelements > 0);
}
