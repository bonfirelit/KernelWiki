---
id: migration-cuda-to-hip
title: "Migrating a CUDA kernel to HIP"
type: migration
from_arch: sm90
to_arch: gfx942
tags: [hip, wmma, mfma, lds, wave64, wave32, tma, global-load-lds]
related: [lang-hip, lang-cuda-cpp, hw-mfma-cdna, hw-wmma-rdna4, hw-lds, hw-amd-memory-ops, migration-cdna-to-rdna4]
sources: [doc-amd-rdna4-matrix-cores, doc-rocm-workload-optimization, doc-llvm-amdgpu-usage, blog-salykova-matrix-cores-cdna, blog-rocm-memory-scheduling]
blackwell_relevance: "Frames the AMD lane against the Hopper/Blackwell mechanisms this wiki documents in depth (wgmma/tcgen05, TMA, mbarrier, TMEM), so a reader arriving from the NVIDIA lane can see which of those have counterparts and which do not."
amd_relevance: "Entry point into the AMD lane for readers whose mental model is CUDA. Targets gfx942 as the CDNA landing point; migration-cdna-to-rdna4 continues to gfx1201."
confidence: source-reported
reproducibility: pseudocode
---

# Migrating a CUDA kernel to HIP

`hipify` converts the API calls. It does not convert the parts that matter. The
mechanical translation compiles and runs; the performance work is where the two
architectures stop resembling each other.

## What hipify handles

`cudaMalloc` -> `hipMalloc`, `__syncthreads()` unchanged, `blockIdx`/`threadIdx`
unchanged, `<<<>>>` or `hipLaunchKernelGGL`. Assume this part is free.

## Required changes

| CUDA / Hopper-Blackwell | HIP / CDNA | Note |
|---|---|---|
| `warpSize` == 32 always | **64 on CDNA**, 32 on RDNA | The highest-yield bug class. Reductions, ballots, and `__shfl` masks written around a literal 32 are wrong on gfx942. Read `warpSize`; do not assume. |
| `wmma::` fragment API / `mma.sync` / `wgmma` / `tcgen05.mma` | `__builtin_amdgcn_mfma_*` (CDNA), `__builtin_amdgcn_wmma_*` (RDNA) | No layout-abstracting fragment type. You compute the per-lane index yourself: `M*K/64` A elements per lane on wave64. |
| TMEM accumulators (SM100) | AGPRs (CDNA) / VGPRs (RDNA) | Same intent — keep C off the critical path — different mechanism. |
| TMA descriptors, `cp.async.bulk` | `llvm.amdgcn.raw.buffer.load.lds` on CDNA; nothing on RDNA4 | No descriptor object, no multicast, no tensor-map. |
| `cp.async` + `cp.async.wait_group` | `global_load_dwordx4` + `s_waitcnt vmcnt(n)` | Wait counts are explicit and per-class. |
| `mbarrier` / named barriers | `s_barrier` + `s_waitcnt lgkmcnt(0)` | No arrive/wait split, no phase parity. |
| Shared memory, 228 KB/SM on SM100 | LDS, 64 KiB (CDNA3, RDNA4) or 160 KiB (CDNA4) | Address space 3, and 32-bit — an LDS pointer is not a global pointer. |
| Warp specialization + CLC persistent scheduling | `sched_group_barrier` / `s_setprio` interleaving | AMD has no producer/consumer warp hardware; the analogue is compiler scheduling. |
| PTX inline asm | AMDGCN inline asm | `lang-amdgcn-asm`. |
| `nvcc -arch=sm_90a` | `hipcc --offload-arch=gfx942` | Names from `doc-llvm-amdgpu-usage`. |
| `ncu` / Nsight Compute | `rocprofv3`, ROCm Compute Profiler, ATT | `AMDGCN_ENABLE_DUMP=1` for the ISA. |

## Porting order

1. **Get it correct on wave64 first.** Fix every lane-width assumption before
   touching performance. This is the step that silently produces wrong answers.
2. **Re-derive the tiling around the matrix instruction**, not around the old
   one. MFMA has a shape family with real trade-offs (`hw-mfma-cdna`), and the
   CUDA-side instinct that the largest tile wins is contradicted on AMD:
   `doc-rocm-workload-optimization` reports `mfma_16x16` typically beating
   `mfma_32x32` for GEMM on MI300X.
3. **Rebuild the load path.** There is no TMA. `blog-rocm-memory-scheduling`
   traces the CDNA replacement end to end: HBM -> VGPR -> LDS -> AGPR, with 16
   `buffer_load_dwordx4` per K-tile per lane, `ds_write_b128`/`ds_read_b128`
   staging, and two `s_barrier`s per iteration.
4. **Read the ISA.** The checklist in `lang-amdgcn-asm` — `global_load_dwordx4`,
   `_b128` LDS forms, non-zero `s_waitcnt` operands — catches most of the gap.
5. **Only then** consider scheduling builtins.

## Inverted intuitions worth flagging

- **More pipeline stages is not better.** For a Triton kernel with two fused
  GEMMs (the FlashAttention shape) `doc-rocm-workload-optimization` prescribes
  `num_stages=1`.
- **High occupancy is often not the goal.** `blog-rocm-occupancy-mi355x` measures
  an MFMA-bound kernel holding ~97% of matrix peak down to ~12% occupancy.
- **Vectorizing loads outranks adopting the matrix core** in measured impact:
  11.2x versus 6.3x in `kernel-cdna4-fp8-gemm`'s ladder.

## Continuing to RDNA4

This page lands on CDNA. gfx1201 removes MFMA, AGPRs, direct-to-LDS, and the LDS
transpose read, and halves the wave width again — see
`migration-cdna-to-rdna4`.
