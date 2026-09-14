---
id: technique-instruction-scheduling-amd
title: "Hand-Scheduling AMD Kernels (sched_barrier, sched_group_barrier, s_setprio)"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [instruction-scheduling, sched-barrier, mfma, lds, pipeline-stages]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-amd-memory-ops, hw-lds]
related: [hw-mfma-cdna, hw-amd-memory-ops, technique-ping-pong-scheduling, technique-pipeline-stages, pattern-pipeline-stalls, kernel-cdna4-fp8-gemm]
sources: [blog-rocm-fp8-gemm-cdna4, blog-rocm-memory-scheduling, doc-rocm-workload-optimization]
symptoms: [pipeline-stalls, low-compute-utilization]
---

# Hand-Scheduling AMD Kernels

AMD has no warp-specialization hardware and no `mbarrier`. The analogous lever
is telling the *compiler's* scheduler what interleaving you want, then telling
the *hardware* arbiter which wave should win. Three builtins do the work.

## Why it is needed: lock-stepping

`blog-rocm-memory-scheduling` states the problem precisely. Waves in a
matrix-multiply kernel issue their memory operations in sync, contend for the
same resources, and leave bandwidth unused. Nothing in the default schedule
breaks that symmetry, so the fix is explicit.

## The three builtins

```cpp
// Verified to compile with hipcc --offload-arch=gfx1201 (ROCm 7.2).
__device__ void pipelined_step(const float4* __restrict__ g,
                               float4* __restrict__ lds,
                               float* acc, int i) {
  float4 next = g[i];                            // global_load_dwordx4

  // (1) Hard fence for the scheduler. Nothing crosses it in either direction.
  __builtin_amdgcn_sched_barrier(0);

  // (2) Build an explicit issue pattern. Groups with a matching sync_id are
  //     ordered against each other; `size` is how many instructions of the
  //     masked class join the group. Instructions are selected bottom-up.
  __builtin_amdgcn_sched_group_barrier(/*mask=*/0x0020, /*size=*/1, /*sync_id=*/0);
  __builtin_amdgcn_sched_group_barrier(/*mask=*/0x0002, /*size=*/4, /*sync_id=*/0);

  // (3) Win the arbiter for the math phase, then hand it back. Priority 0..3.
  __builtin_amdgcn_s_setprio(1);
  acc[0] += next.x * next.y;
  __builtin_amdgcn_s_setprio(0);

  lds[i] = next;                                 // ds_write_b128
  __builtin_amdgcn_s_barrier();                  // workgroup barrier
}
```

- **`__builtin_amdgcn_sched_barrier(mask)`** — with `0`, "no instructions may be
  scheduled across `sched_barrier`" (`blog-rocm-fp8-gemm-cdna4`). A non-zero mask
  lets the named instruction classes cross.
- **`__builtin_amdgcn_sched_group_barrier(mask, size, sync_id)`** — forms a
  scheduling group of `size` instructions matching `mask`. Groups sharing a
  `sync_id` are emitted in the order the calls appear, which is how you express
  "one VMEM read, then one VALU, then five MFMA".
- **`__builtin_amdgcn_s_setprio(n)`**, `n` in 0..3 — raises this wave's issue
  priority. `blog-rocm-fp8-gemm-cdna4` brackets its math phase with
  `s_setprio(1)` / `s_setprio(0)`.

## What this is worth

In `blog-rocm-fp8-gemm-cdna4`'s ladder on MI355X, FP8 M=N=K=4096, the
scheduling-driven rungs are the top of the chart: multi-wave `256x256_t512`
reaches 2288.16 TFLOP/s and the 8-wave ping-pong schedule 2680.33, against
hipBLASLt's 2750.42. Instruction scheduling is where the last ~2.3x lives, after
tiling, vectorization, direct-to-LDS, and double buffering are already done. It
is not the first thing to reach for.

## Boundary — read this before copying a mask

The `(mask, size, sync_id)` signature and the `sched_barrier(0)` semantics above
are established by the sources cited on this page. The **specific mask bit
values are not**: no source captured in this wiki publishes the
`SchedGroupMask` table, and the two values in the snippet above are
placeholders chosen to show the shape of the call, not verified class
selectors. A wrong mask silently produces a different schedule and no error —
the kernel still computes the right answer, just not at the speed you measured
elsewhere.

Before relying on a mask, read the `SchedGroupMask` enum in the LLVM tree you
are actually compiling against (`llvm/lib/Target/AMDGPU/AMDGPUIGroupLP.cpp`, and
the `amdgpu.sched_barrier` entry in the MLIR AMDGPU dialect docs), then confirm
against the emitted ISA with `AMDGCN_ENABLE_DUMP=1` that the instruction order
is the one you asked for.

## Transfers to RDNA4?

**Partially.** All three builtins compile for gfx1201 — verified on this host.
But the payoff measured above comes from interleaving MFMA against
direct-to-LDS traffic, and RDNA4 has neither. On RDNA4 the useful pattern is
interleaving `global_load_dwordx4` against WMMA and `ds_write_b128`; expect a
smaller win, and measure rather than porting a CDNA group sequence verbatim.
