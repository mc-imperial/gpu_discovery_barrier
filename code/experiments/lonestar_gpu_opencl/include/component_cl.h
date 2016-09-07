// The device part of the component space for the mst GPU-Lonestar
// application

// OpenCL port by Tyler Sorensen (2016)

#pragma once

typedef struct {
  __global uint *ncomponents, *complen, *ele2comp;
} ComponentSpace;

kernel void init_compspace(__global ComponentSpace *cs, __global uint *ncomponents, __global uint *complen, __global uint *ele2comp, uint nelements) {

  int id = get_global_id(0);

  if (id < nelements) {
    complen[id] = 1;
    ele2comp[id] = id;
  }
  if (id == 0) {
    *ncomponents = nelements;
    cs->ncomponents = ncomponents;
    cs->complen = complen;
    cs->ele2comp = ele2comp;
  }
}

unsigned cs_isBoss(__global ComponentSpace *cs, unsigned element) {
  return atomic_cmpxchg(&(cs->ele2comp[element]), element, element) == element;
}

unsigned cs_find(__global ComponentSpace *cs, unsigned lelement) {
  unsigned element = lelement;
  while (cs_isBoss(cs, element) == 0) {
    element = cs->ele2comp[element];
  }
  cs->ele2comp[lelement] = element;
  return element;
}

int cs_unify(__global ComponentSpace *cs, unsigned one, unsigned two) {

  // If the client makes sure that one component is going to get
  // unified as a source with another destination only once, then
  // synchronization is unnecessary.  while this is true for MST, due
  // to load-balancing in if-block below, a node may be source
  // multiple times.  if a component is source in one thread and
  // destination is another, then it is okay for MST.
  do {
    if(!cs_isBoss(cs, one)) return 0;
    if(!cs_isBoss(cs, two)) return 0;

    unsigned onecomp = one;
    unsigned twocomp = two;

    if (onecomp == twocomp) return 0; // "duplicate" edges due to symmetry

    unsigned boss = twocomp;
    unsigned subordinate = onecomp;

    if (boss < subordinate) { // Break cycles by id.
      boss = onecomp;
      subordinate = twocomp;
    }

    unsigned oldboss = atomic_cmpxchg(&(cs->ele2comp[subordinate]), subordinate, boss);
    if (oldboss != subordinate) { // Someone else updated the boss.

      // We need not restore the ele2comp[subordinate], as union-find
      // ensures correctness and complen of subordinate doesn't
      // matter.
      one = oldboss;
      two = boss;
      return 0;
    }
    else {

      atomic_add(&(cs->complen[boss]), cs->complen[subordinate]);

      // A component has reduced.
      unsigned ncomp = atomic_sub(cs->ncomponents, 1);
      return 1;
    }
  } while (1);
}
