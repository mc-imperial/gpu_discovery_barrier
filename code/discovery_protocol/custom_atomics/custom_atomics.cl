// Custom OpenCL 2.0 atomics implemented for systems that don't support OpenCL 2.0

// Currently only implemented for Nvidia and ARM
// There are no proofs, but these seem to work.

// We support:
// load(relaxed, device
// store(release, device)
// memfence(acquire)
// atomic_fetch_add_explicit(seq_cst, device)
// exchange(seq_cst, device)

#pragma once

typedef int atomic_int;

typedef enum {memory_scope_device} memory_scope;
typedef enum {memory_order_relaxed, memory_order_release, memory_order_acquire, memory_order_acq_rel} memory_order;

// Only use add in sequential consistency mode for device scope
int atomic_fetch_add_explicit(__global volatile atomic_int* target, int operand, memory_order mo, memory_scope ms) {

  int ret = 0;

#ifdef ARM
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif

#ifdef NVIDIA
  asm volatile ("membar.gl;\n");
#endif

  ret =  atomic_add(target, operand);

#ifdef NVIDIA
  asm volatile ("membar.gl;\n");
#endif

#ifdef ARM
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif

  return ret;
}

// We only do store release at the device scope so just do those fences
void atomic_store_explicit(__global volatile atomic_int* target, int val, const memory_order mo, const memory_scope ms) {

#ifdef ARM
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif

#ifdef NVIDIA
  asm volatile ("membar.gl;\n");
#endif

  *target = val;
}

// We only do load relaxed at the device scope, so these fences should be good. We include a memfence
// because some Nvidia chips show CoRR behaviours.
int atomic_load_explicit(__global volatile atomic_int* target, const memory_order mo, const memory_scope ms) {

  int ret = 0;
  ret = *target;

#ifdef NVIDIA
  if (mo == memory_order_acquire) {
    asm volatile ("membar.gl;\n");
  }
#endif

#ifdef ARM
  if (mo == memory_order_acquire) {
    mem_fence(CLK_GLOBAL_MEM_FENCE);
  }
#endif

  return ret;
}

// We only do acquire fences at the device scope, so these fences should be good.
void atomic_work_item_fence(const cl_mem_fence_flags flags, const memory_order mo, const memory_scope ms) {
#ifdef NVIDIA
  asm volatile ("membar.gl;\n");
#endif

#ifdef ARM
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif
}

// Only used with seq_cst at the device scope. These fences should be okay.
int atomic_exchange_explicit(__global volatile atomic_int* target, const int desired, const memory_order mo, const memory_scope ms) {
  int old = 0;

#ifdef NVIDIA
  asm volatile ("membar.cta;\n");
#endif

#ifdef ARM
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif

  old = atomic_xchg(target, desired);

#ifdef NVIDIA
  asm volatile ("membar.gl;\n");
#endif

#ifdef ARM
  mem_fence(CLK_GLOBAL_MEM_FENCE);
#endif

  return old;
}
