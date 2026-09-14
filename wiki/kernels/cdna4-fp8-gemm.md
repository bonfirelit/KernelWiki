---
id: kernel-cdna4-fp8-gemm
title: "FP8 GEMM on CDNA4 — the measured optimization ladder"
type: kernel
architectures: [gfx950, cdna4]
tags: [gemm, mfma, ocp-fp8, lds, global-load-lds, double-buffering, swizzling, lds-bank-conflict-avoidance, instruction-scheduling, ping-pong-scheduling, vectorized-loads]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [hip]
related: [hw-mfma-cdna, hw-amd-memory-ops, hw-lds, hw-amd-narrow-precision, technique-lds-bank-conflict-avoidance, technique-instruction-scheduling-amd, technique-double-buffering, technique-ping-pong-scheduling, migration-cdna-to-rdna4]
sources: [blog-rocm-fp8-gemm-cdna4, blog-salykova-matrix-cores-cdna]
performance_claims:
  - gpu: "MI355X (gfx950)"
    dtype: "FP8 E4M3FN in, BF16 out, FP32 accumulate"
    shape: "M=N=K=4096"
    metric: TFLOPS
    value: 2680.33
    utilization: "97% of hipBLASLt's 2750.42 on the same shape"
    source_id: blog-rocm-fp8-gemm-cdna4
    source_locator: "'8-Wave Ping-Pong' row of the optimization ladder table, ROCm 7.1.0"
  - gpu: "MI355X (gfx950)"
    dtype: "FP8 E4M3FN in, BF16 out, FP32 accumulate"
    shape: "M=N=K=8192"
    metric: TFLOPS
    value: 3204.15
    utilization: "ahead of hipBLASLt's 3130.21 on the same shape"
    source_id: blog-rocm-fp8-gemm-cdna4
    source_locator: "'8-Wave Ping-Pong' result at M=N=K=8192, ROCm 7.1.0"
---

# FP8 GEMM on CDNA4 — the measured optimization ladder

The most useful single AMD kernel study available, because every rung is
measured on one shape. It is the AMD-side analogue of `kernel-deepgemm`, and the
per-rung deltas are the evidence base for the ordering advice throughout this
wiki's AMD pages.

## The ladder

MI355X (gfx950), ROCm 7.1.0, FP8 E4M3FN inputs, BF16 output, FP32 accumulate,
M=N=K=4096, TFLOP/s (`blog-rocm-fp8-gemm-cdna4`):

| Rung | TFLOP/s | Delta | What it added |
|---|---:|---|---|
| Naive | 1.15 | — | |
| LDS tiling | 4.80 | 4.2x | data reuse through `hw-lds` |
| Matrix-core baseline | 30.05 | 6.3x | MFMA instead of VALU FMA |
| Vectorized load | 336.88 | **11.2x** | `global_load_dwordx4` |
| Direct global-to-LDS | 506.70 | 1.5x | `llvm.amdgcn.raw.buffer.load.lds` |
| LDS swizzle | 497.43 | 0.98x | XOR remap — *no win yet* |
| Double buffering | 1166.41 | **2.3x** | overlap next tile with current MFMA |
| Multi-wave 256x256_t512 | 2288.16 | 2.0x | bigger tile, more waves |
| 8-wave ping-pong | 2680.33 | 1.2x | `sched_barrier` + `s_setprio` |
| hipBLASLt | 2750.42 | — | reference |

At M=N=K=8192 the same kernel reaches 3204.15 against hipBLASLt's 3130.21.

## Reading the deltas

Three things worth internalizing:

1. **Vectorization is the single largest step** — 11.2x, larger than adopting the
   matrix core. Getting `global_load_dwordx4` is the cheapest work on the list,
   and it is the first item on the ISA checklist in `hw-amd-memory-ops` for
   exactly this reason.
2. **The LDS swizzle is a regression in isolation** (0.98x) and becomes a win only
   after double buffering raises the LDS pressure it relieves. Optimizations are
   not independently additive; see the boundary note in
   `technique-lds-bank-conflict-avoidance`.
3. **Scheduling is last and worth 1.2x.** Reaching for `sched_barrier` before
   vectorizing is a 10x mistake in priority.

## Verbatim anchor

The rungs above correspond to concrete, named mechanisms:

```cpp
// Rung 5: direct global-to-LDS. CDNA4 widens the per-lane GLOBAL_LOAD_LDS
// transfer to 128 bits/lane (32-bit on CDNA3), so the tile never lands in VGPRs.
llvm_amdgcn_raw_buffer_load_lds(/* rsrc */ a_desc, /* lds */ a_lds_offset,
                                /* size */ 16, /* voffset */ voff,
                                /* soffset */ 0, /* offset */ 0,
                                /* aux */ 0);

// Rung 6: XOR swizzle on 16-byte columns. ds_read_b128 runs in four phases and
// each must be conflict-free.
const int swizzled_col = c ^ (perm << 4);

// Rung 9: the ping-pong schedule. Fence the scheduler, then win the arbiter
// for the math phase and hand it straight back.
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
/* MFMA cluster */
__builtin_amdgcn_s_setprio(0);
__builtin_amdgcn_s_barrier();
```

Instruction mix: the `16x16x128` FP8 MFMA form with FP32 accumulation, output
tiles of 128x128 and 256x256, K-tile 128. CDNA4's block-scaled additions are
`V_MFMA_SCALE_F32_16X16X128_F8F6F4` and `V_MFMA_SCALE_F32_32X32X64_F8F6F4`.

## Performance boundary

Two square shapes on one GPU under one ROCm version. Nothing here establishes
the ordering or the magnitudes for skinny GEMMs, for MoE grouped shapes, or for
BF16 — and `blog-rocm-mxfp4-rotation` shows tile selection inverting at M=1,
where the smaller `16x16x32` form beats `32x32x16` precisely because the large
tile underuses the matrix core.

## Transfers to RDNA4?

**Rungs 1-4 and 7-9 transfer; rungs 5-6 do not.** gfx1201 has no direct-to-LDS
path, so the tile must route HBM -> VGPR -> LDS, holding VGPRs for the duration
and pushing toward smaller K-tiles. It also has no MFMA — WMMA's fixed 16x16x16
replaces the whole shape-selection dimension. `migration-cdna-to-rdna4` is the
full accounting.
