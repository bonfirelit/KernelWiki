---
id: technique-mfma-pipelining
title: "MFMA Software Pipelining — Keeping the Matrix Core Fed"
type: technique
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [mfma-pipelining, mfma, sched-barrier, double-buffering, lds]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma-cdna, technique-instruction-scheduling-amd]
related: [technique-occupancy-tuning-amd, technique-lds-bank-conflict-avoidance, kernel-cdna4-fp8-gemm, hw-lds]
sources: [blog-rocm-fp8-gemm-cdna4, blog-rocm-memory-scheduling, blog-rocm-occupancy-mi355x, blog-rocm-kernel-wiki, doc-cdna3-isa]
symptoms: [low-compute-utilization, pipeline-stalls]
---

# MFMA Software Pipelining

The matrix unit is a **separate issue port** from VALU and memory. A
`v_mfma_f32_16x16x16_f16` occupies it for ~16 cycles while the issuing lane is
free to issue loads (`blog-rocm-kernel-wiki`). Stall on a wait until the next
K-slice lands and the matrix unit idles for the whole memory latency.
Diagnose with the `MfmaUtil` path in `technique-occupancy-tuning-amd`: on
MI355X an MFMA-bound sweep read ~70% at ILP=1 against ~98% at ILP=8
(`blog-rocm-occupancy-mi355x`). That gap is what pipelining recovers.

CDNA has no `mbarrier` and no async proxy. Overlap is built from relaxed
`s_waitcnt` targets (same-type memory ops retire in issue order, so `vmcnt(N)`
means "all but the last N issued VMEM ops have landed" (`doc-cdna3-isa`)) plus
scheduling fences when the compiler collapses the interleave.

## Pattern 1 — prefetch one K-slice ahead, gate with a relaxed count

Issue slice k+1's loads **before** consuming slice k, then wait only deep
enough that slice k's data has landed:

```cpp
#include <hip/hip_runtime.h>
using f16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// Register-pipelined K loop: slice k+1's loads issue before slice k's MFMA;
// the wait leaves the two fresh loads outstanding.
__device__ f32x4 gemm_k_loop(const f16x4* __restrict__ A,
                             const f16x4* __restrict__ B, int k_tiles) {
  const int lane = threadIdx.x & 63;
  f16x4 frag[2][2];
  frag[0][0] = A[lane]; frag[0][1] = B[lane];        // prologue: slice 0
  f32x4 acc = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < k_tiles; ++k) {
    const int cur = k & 1, nxt = cur ^ 1;
    if (k + 1 < k_tiles) {                           // prefetch slice k+1
      frag[nxt][0] = A[(k + 1) * 64 + lane];
      frag[nxt][1] = B[(k + 1) * 64 + lane];
    }
    __builtin_amdgcn_s_waitcnt(2);  // <=2 VMEM loads out: slice k has landed
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(frag[cur][0], frag[cur][1],
                                                acc, 0, 0, 0);
  }
  return acc;
}
```

One encoding trap, verified on this host (ROCm 7.2.4, gfx950): the builtin's
argument is the **raw SIMM16 immediate**, so `__builtin_amdgcn_s_waitcnt(2)`
emits `s_waitcnt vmcnt(2) expcnt(0) lgkmcnt(0)` — the LDS counter is pinned to
zero too. Fine in a VMEM-only loop like this one; with LDS prefetches in
flight it is a hidden full drain — use `asm volatile("s_waitcnt vmcnt(2)")`.
Never hand-compute hex immediates (`doc-cdna3-isa`).

## The double-buffer discipline — two barriers, not one

When the prefetch stages through LDS, each iteration needs **two** `s_barrier`s
(`blog-rocm-memory-scheduling`):

```text
s_barrier              # read-drain: no wave still reads the buffer
                       # this iteration's stores will overwrite
ds_write  next tile -> LDS buf[nxt]
s_waitcnt lgkmcnt(0)   # this wave's stores have landed
s_barrier              # write-drain: stores visible before any wave reads
ds_read buf[cur] -> v_mfma ...              # reads gated with lgkmcnt(N)
global_load tile k+2 -> VGPR                # gated later with vmcnt(N)
```

A single barrier after the stores fails twice: a fast wave can lap a slow one
and overwrite LDS the slow wave still reads, and the `vmcnt(0)` usually paired
with it drains the in-flight prefetch. Keep waits at `vmcnt(N)`/`lgkmcnt(N)`.

## Pattern 2 — pin the interleave with sched_barrier

The LLVM scheduler will happily hoist your MFMA block above the loads that
feed it, collapsing the overlap. `__builtin_amdgcn_sched_barrier(mask)` is a
compile-time fence with **this polarity** (LLVM's `AMDGPUIGroupLP`): set mask
bits name the instruction classes **allowed to cross**; `mask=0` blocks
everything — the form `blog-rocm-fp8-gemm-cdna4` uses around its math phase.
For explicit issue patterns ("one VMEM read, then five MFMAs") use
`sched_group_barrier`. `technique-instruction-scheduling-amd` has the builtin
inventory; heed its warning that no captured source publishes the mask-bit
table — read `SchedGroupMask` out of your LLVM tree.

## Pattern 3 — interleave waves so one issues MFMA while others stream

One wave's ILP has a ceiling; the complementary lever is resident waves per
SIMD. The schedule in `blog-rocm-fp8-gemm-cdna4` interleaves waves so that at
any cycle roughly **one wave per SIMD is issuing MFMA while the others stream
operands** — the shape matters more than the count. Provision residency with
the occupancy math in `technique-occupancy-tuning-amd` (accumulator AGPRs and
double-buffered LDS are what you spend) rather than trusting a magic number.

In the `kernel-cdna4-fp8-gemm` ladder this is the **last** rung: scheduling
was worth 1.2x (2288 → 2680 TFLOP/s) only after vectorization (11.2x) and
double buffering (2.3x) were in place (`blog-rocm-fp8-gemm-cdna4`). Reaching
for `sched_barrier` before those is a priority error.

## Transfers to RDNA4?

**Patterns 1 and 2 transfer as ideas; Pattern 3 does not.** gfx1201 has no
MFMA — the fixed 16x16x16 WMMA on wave32 replaces it, and a wave interleave
built around MFMA issue slots has no analogue. Prefetch-ahead with relaxed
`vmcnt(N)` and the scheduling builtins compile for gfx1201 (verified in
`technique-instruction-scheduling-amd`); the useful interleave there is
`global_load` against WMMA and `ds_write`. Expect a smaller win; re-measure.
