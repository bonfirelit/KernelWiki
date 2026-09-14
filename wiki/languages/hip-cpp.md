---
id: lang-hip
title: "HIP C++ for AMD kernels"
type: language
tags: [hip, wmma, mfma, lds, amdgcn-asm]
related: [hw-gfx1201, hw-wmma-rdna4, hw-mfma-cdna, hw-lds, hw-amd-memory-ops, lang-amdgcn-asm, lang-triton-rocm, lang-composable-kernel, migration-cuda-to-hip]
sources: [doc-amd-rdna4-matrix-cores, doc-rocm-workload-optimization, doc-llvm-amdgpu-usage, blog-salykova-matrix-cores-cdna]
reproducibility: snippet
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
confidence: source-reported
aliases: [HIP, "HIP C++", hipcc]
---

# HIP C++ for AMD kernels

HIP is the first-class device language on ROCm: a C++ dialect that is
deliberately close to CUDA C++ at the API surface, and deliberately *not* close
at the instruction surface. Matrix instructions, LDS transposes, wait counters,
and scheduling hints are all reached through `__builtin_amdgcn_*` compiler
builtins rather than a portable abstraction.

## The compile boundary

```bash
# Compile for exactly the target you will run on. gfx12-generic also exists but
# gives up target-specific instruction selection.
hipcc --offload-arch=gfx1201 kernel.hip -o kernel

# Multiple targets in one fat binary.
hipcc --offload-arch=gfx942 --offload-arch=gfx950 --offload-arch=gfx1201 k.hip -o k

# Read what the compiler actually emitted --- this is not optional work.
AMDGCN_ENABLE_DUMP=1 ./kernel 2>&1 | grep -E '\.vgpr_count|\.agpr_count|v_(wmma|mfma)|ds_read|s_waitcnt'
```

`--offload-arch` values are LLVM AMDGPU processor names; `doc-llvm-amdgpu-usage`
carries the product-to-processor table.

## Verbatim call-site excerpt

Matrix-core access is a builtin call from every lane of the wavefront. Compiled
on this host with `hipcc --offload-arch=gfx1201` (ROCm 7.2):

```cpp
#include <hip/hip_runtime.h>

typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));

__global__ void wmma_gemm_16x16(const _Float16* __restrict__ A,
                                const _Float16* __restrict__ B,
                                float* __restrict__ D) {
  const int lane = threadIdx.x;        // wave32: 0..31
  const int col  = lane % 16;
  const int rb   = (lane / 16) * 8;

  half8  a, b;
  float8 c = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

  for (int j = 0; j < 8; ++j) {
    a[j] = A[(rb + j) * 16 + col];     // A column-major
    b[j] = B[(rb + j) * 16 + col];     // B row-major
  }

  c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);

  for (int j = 0; j < 8; ++j)
    D[(rb + j) * 16 + col] = c[j];
}
```

## What differs from CUDA C++ in practice

| Concern | CUDA C++ | HIP C++ |
|---|---|---|
| Wave width | 32, fixed | 32 on RDNA, **64 on CDNA** — `warpSize` is not a constant you may assume |
| Matrix core | `wmma::` API / `mma.sync` PTX | `__builtin_amdgcn_wmma_*` / `__builtin_amdgcn_mfma_*`, no layout-abstracting fragment type |
| Async copy | `cp.async` / TMA descriptors | `llvm.amdgcn.raw.buffer.load.lds` on CDNA only |
| Completion | `mbarrier` | `s_waitcnt vmcnt/lgkmcnt` + `s_barrier` |
| Accumulator storage | registers, or TMEM on SM100 | registers; AGPRs on CDNA |
| Inline asm | PTX | AMDGCN — see `lang-amdgcn-asm` |

The wave-width difference is the one that silently breaks ported code: a
reduction written around a hard-coded 32 is wrong on gfx942 and a `__shfl` mask
built from `0xFFFFFFFF` covers half a CDNA wavefront. `migration-cuda-to-hip`
covers the rest.

## Selection guidance

Reach for HIP C++ when you need a specific matrix instruction, a specific LDS
layout, or hand-placed wait counts — that is, when the schedule *is* the
optimization. Prefer `lang-triton-rocm` when the kernel is shape-parametric and
autotuning over tile sizes will find most of the win, and
`lang-composable-kernel` when a templated CDNA GEMM/attention pipeline already
exists for your shape.
