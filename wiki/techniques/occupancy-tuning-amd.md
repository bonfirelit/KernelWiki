---
id: technique-occupancy-tuning-amd
title: "Occupancy Tuning on AMD (and when not to)"
type: technique
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
tags: [occupancy-tuning, lds, agpr, mfma, register-budgeting, triton-rocm]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-lds]
related: [hw-lds, hw-gfx1201, hw-mfma-cdna, technique-register-budgeting, pattern-register-pressure, pattern-low-sm-utilization, lang-triton-rocm]
sources: [doc-rocm-workload-optimization, blog-rocm-occupancy-mi355x]
symptoms: [register-pressure, low-occupancy, low-compute-utilization]
---

# Occupancy Tuning on AMD (and when not to)

Occupancy on AMD is the minimum of four independent limiters, and the headline
finding of `blog-rocm-occupancy-mi355x` is that for the kernels you most want to
tune, maximizing it is the wrong goal.

## The four limiters

Per `blog-rocm-occupancy-mi355x` (MI355X / CDNA4), occupancy is the minimum of:

| Limiter | Expression |
|---|---|
| VGPR | `floor(512 / VGPRs-per-lane)` waves/SIMD |
| SGPR | `floor(~800 / SGPRs-per-wave)` waves/SIMD |
| LDS | `floor(160 KB / LDS-per-workgroup)` workgroups/CU |
| Workgroup | hardware resident-workgroup cap and barrier slots |

clamped to 8 waves/SIMD and 32 waves/CU. On CDNA4 VGPRs and AGPRs share one
512-register-per-lane file, with neither class exceeding 256.

**Allocation granularity is per-generation and it matters.**
`doc-rocm-workload-optimization` states MI300X allocates VGPRs "in blocks of 16"
with a worked example: usage of 170 rounds to 176, and `176 x 3 > 512` caps
occupancy at 2 waves/EU. `blog-rocm-occupancy-mi355x` reports MI355X rounding "to
groups of 8". Do not carry a granularity constant across targets.

## Computing it for a real kernel

```bash
# 1. VGPR count, straight from the ISA the compiler actually emitted.
AMDGCN_ENABLE_DUMP=1 python my_kernel.py 2>&1 | grep -E '\.vgpr_count|\.agpr_count|\.sgpr_count'

# 2. LDS bytes per workgroup and waves per workgroup, from the Triton MLIR dump.
MLIR_ENABLE_DUMP=1 python my_kernel.py 2>&1 | grep -E 'triton_gpu.shared|triton_gpu.num-warps'

# 3. occ_lds = floor(65536 / L)   on 64 KiB parts (MI300X, gfx1201)
#    occ_lds = floor(163840 / L)  on MI350X/MI355X
#    occ     = min(floor(occ_vgpr * 4 / nW), occ_lds) * nW / 4
```

`occ_vgpr * 4` is the wave count across all four SIMDs of a CDNA CU; the final
occupancy is the smaller of the register- and LDS-derived figures
(`doc-rocm-workload-optimization`).

## The one knob worth reaching for

`waves_per_eu=n` asks the LLVM backend to shrink VGPR usage until `n` waves fit.
It is worth trying in exactly one situation: occupancy is VGPR-limited **and**
usage sits a few registers above a granularity boundary. At 176 registers, asking
for 3 waves/EU may cost nothing. At 250 it will cost spills.

```python
# Triton on ROCm. num_stages: 2 for a single GEMM, 1 for two fused GEMMs
# (Flash-Attention shaped), 2 for a GEMM fused with a non-GEMM op, 1 with no GEMM.
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...

gemm_kernel[grid](
    a, b, c, M, N, K,
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
    num_stages=2,
    num_warps=4,
    waves_per_eu=2,              # only after confirming a VGPR-limited kernel
    matrix_instr_nonkdim=16,     # 16 -> mfma_16x16; 32 -> mfma_32x32
)
```

## Why high occupancy is often the wrong target

`blog-rocm-occupancy-mi355x` measured an MXFP8 MFMA sweep on MI355X. With ILP=8
the matrix core held 4.82-4.84 PFLOP/s — "~97% of the MI355X GPU's ~5 PFLOP/s
MXFP8 matrix peak" — **all the way down to ~12% occupancy**. At that floor
`MfmaUtil` read ~98% for ILP=8 against ~70% for ILP=1. The conclusion stated
there: for MFMA-bound kernels "the sweet spot is routinely 2-4 waves/SIMD,
not 8."

The mechanism is that throughput tracks matrix-engine utilization, not how full
the SIMD is. Instruction-level parallelism within one wave — several independent
MFMAs in flight — substitutes for wave-level parallelism. Chasing occupancy by
shrinking tiles reduces per-wave ILP and can cost more than the extra waves
recover.

So: use the occupancy math to find out **which** resource is binding, then decide
whether that resource is actually on the critical path. If the kernel is
MFMA-bound at 12% occupancy, the answer is to leave it alone and go look at
`technique-instruction-scheduling-amd` instead.

## Transfers to RDNA4?

**The method, yes; the constants, no.** gfx1201 has 64 KiB of LDS per CU (use
the 65536 divisor), 2 SIMDs per CU rather than 4 — so the `* 4` in the CDNA
formula is wrong there — and 32 max waves per CU. And with no MFMA on RDNA4, the
"MFMA-bound kernels don't need occupancy" finding does not transfer as stated;
re-measure against WMMA before assuming it holds.
