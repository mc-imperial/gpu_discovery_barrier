// Simple lock implementations to use in the discovery protocol.
// Either an unfair spin lock can be used or a fair ticket lock.

#pragma once

#ifdef CUSTOM_ATOMICS
#include "../custom_atomics/custom_atomics.cl"
#endif

// Spin lock (unfair)
#ifdef SPIN_LOCK

void discovery_lock(__global discovery_mutex *m) {

  while(atomic_exchange_explicit(&(m->counter), 1, memory_order_acq_rel, memory_scope_device) == 1);
}

void discovery_unlock(__global discovery_mutex *m) {
  atomic_store_explicit(&(m->counter), 0, memory_order_release, memory_scope_device);
}

// Ticket lock (fair)
#else

void discovery_lock(__global discovery_mutex *m) {
  int ticket = atomic_fetch_add_explicit(&(m->counter), 1, memory_order_acq_rel, memory_scope_device);
  while (atomic_load_explicit(&(m->now_serving), memory_order_acquire, memory_scope_device) != ticket);
}

void discovery_unlock(__global discovery_mutex *m) {
  int tmp = atomic_load_explicit(&(m->now_serving), memory_order_acquire, memory_scope_device);
  tmp+=1;
  atomic_store_explicit(&(m->now_serving), tmp, memory_order_release, memory_scope_device);
}

#endif
