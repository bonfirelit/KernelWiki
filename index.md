# GPU Kernel Optimization Knowledge Base

> Comprehensive knowledge base for GPU kernel optimization across two vendor lanes:
> NVIDIA Blackwell (SM100) / Hopper (SM90), and AMD CDNA3/CDNA4 / RDNA3/RDNA4.
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and conventions.
> **For Claude Code agents**: this repository is a Claude Code skill — see [SKILL.md](SKILL.md).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for 10 worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Browse exact, family-only, or unknown architecture evidence | [queries/by-architecture.md](queries/by-architecture.md) |
| Work on AMD (gfx1201 / MI300X / MI355X) | [AMD lane](#amd-lane-cdna34-rdna34) below |
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |

## NVIDIA lane (Blackwell / Hopper)

### Hardware Features

- [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md) — Blackwell MMA instruction (replaces wgmma)
- [hw-tmem](wiki/hardware/tmem.md) — Tensor Memory (CTA-visible 128-lane × 512-column view)
- [hw-clc](wiki/hardware/clc.md) — Cluster Launch Control (dynamic tile scheduling)
- [hw-tma](wiki/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads)
- [hw-2sm-cooperative](wiki/hardware/2sm-cooperative.md) — Two-SM cooperative MMA
- [hw-nvfp4](wiki/hardware/nvfp4.md) — NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — Programmatic Dependent Launch / Grid Dependency Control

### Optimization Techniques

- [technique-warp-specialization](wiki/techniques/warp-specialization.md) — Warp role assignment
- [technique-persistent-kernels](wiki/techniques/persistent-kernels.md) — Persistent kernel patterns with CLC
- [technique-swizzling](wiki/techniques/swizzling.md) — Shared memory swizzling
- [technique-pipeline-stages](wiki/techniques/pipeline-stages.md) — Software pipelining
- [technique-epilogue-fusion](wiki/techniques/epilogue-fusion.md) — Fusing epilogue with mainloop
- [technique-tile-scheduling](wiki/techniques/tile-scheduling.md) — Tile scheduling strategies
- [technique-double-buffering](wiki/techniques/double-buffering.md) — Double/multi-buffering
- [technique-software-exp](wiki/techniques/software-exp.md) — Software-emulated exponential
- [technique-fine-grained-quantization](wiki/techniques/fine-grained-quantization.md) - Fine-grained FP8/FP4 quantization
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md) — Wide vectorized loads and cache policies

### Kernel Case Studies

- [kernel-flash-attention-4](wiki/kernels/flash-attention-4.md) — FlashAttention-4 (up to 1613 TFLOPS on B200 in the paper's benchmark sweep)
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — DeepGEMM FP8 GEMM (1550 TFLOPS on H800)
- [kernel-flashmla](wiki/kernels/flashmla.md) — FlashMLA sparse/dense MLA decoding
- [kernel-nsa](wiki/kernels/nsa.md) — Native Sparse Attention (9x fwd speedup)
- [kernel-gated-delta-net](wiki/kernels/gated-delta-net.md) — Gated Delta Net linear attention
- [kernel-nvfp4-gemm](wiki/kernels/nvfp4-gemm.md) — NVFP4 GEMM from GPU Mode hackathon
- [kernel-nvfp4-gemv](wiki/kernels/nvfp4-gemv.md) — NVFP4 batched GEMV optimization
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — Fused MoE with FP8

### Problem → Solution Patterns

- [pattern-low-sm-utilization](wiki/patterns/low-sm-utilization.md) — SM utilization is low
- [pattern-memory-bound](wiki/patterns/memory-bound.md) — Kernel is memory bandwidth limited
- [pattern-register-pressure](wiki/patterns/register-pressure.md) — Too many registers → low occupancy
- [pattern-compute-bound](wiki/patterns/compute-bound.md) — Not reaching peak FLOPS
- [pattern-tail-effect](wiki/patterns/tail-effect.md) — Last wave underutilizes GPU

### Languages & DSLs

- [lang-cute-dsl](wiki/languages/cute-dsl.md) — CuTe DSL for Blackwell
- [lang-cuda-cpp](wiki/languages/cuda-cpp.md) — CUDA C++ with PTX inline
- [lang-ptx](wiki/languages/ptx-sm100.md) — PTX instructions for SM100
- [lang-triton](wiki/languages/triton-blackwell.md) — Triton on Blackwell

### Migration Guides

- [migration-wgmma-to-tcgen05](wiki/migration/wgmma-to-tcgen05.md) — Hopper wgmma → Blackwell tcgen05
- [migration-register-to-tmem](wiki/migration/register-to-tmem.md) — Register accumulators → TMEM

## AMD lane (CDNA3/4, RDNA3/4)

RDNA4-first — **gfx1201 is the depth target**, and every CDNA page states whether
its content transfers to RDNA4.

### Hardware Features

- [hw-gfx1201](wiki/hardware/gfx1201.md) — RDNA4 as a kernel target; wave32, 64 KiB LDS, WGP-vs-CU counting
- [hw-wmma-rdna4](wiki/hardware/wmma-rdna4.md) — WMMA on gfx12; the fragment lane mapping and the silent-transpose trap
- [hw-mfma-cdna](wiki/hardware/mfma-cdna.md) — MFMA shape family, AGPR accumulators, CDNA4 block scaling
- [hw-lds](wiki/hardware/lds.md) — Local Data Share: capacity, banking, `lgkmcnt`
- [hw-amd-memory-ops](wiki/hardware/amd-memory-ops.md) — `global_load_dwordx4`, direct-to-LDS, wait counters
- [hw-amd-narrow-precision](wiki/hardware/amd-narrow-precision.md) — OCP FP8, MXFP4/MXFP8, E8M0 scales

### Optimization Techniques

- [technique-lds-bank-conflict-avoidance](wiki/techniques/lds-bank-conflict-avoidance.md) — XOR swizzle vs padding
- [technique-instruction-scheduling-amd](wiki/techniques/instruction-scheduling-amd.md) — `sched_barrier`, `sched_group_barrier`, `s_setprio`
- [technique-occupancy-tuning-amd](wiki/techniques/occupancy-tuning-amd.md) — the four limiters, and when occupancy is the wrong target
- [technique-in-register-transpose](wiki/techniques/in-register-transpose.md) — transposing WMMA operands without `ds_read_tr`

### Kernel Case Studies

- [kernel-cdna4-fp8-gemm](wiki/kernels/cdna4-fp8-gemm.md) — FP8 GEMM on MI355X: the measured ten-rung ladder (2680 TFLOP/s at 4096³)
- [kernel-rdna4-wmma-gemm](wiki/kernels/rdna4-wmma-gemm.md) — fused MXFP4→FP16 WMMA GEMM on gfx1201 (40.8 TFLOPS, 53% of theoretical)

### Problem → Solution Patterns

- [pattern-lds-bank-conflicts](wiki/patterns/lds-bank-conflicts.md) — LDS accesses serialize
- The NVIDIA pattern pages above also carry `## On AMD (CDNA / RDNA4)` sections

### Languages & DSLs

- [lang-hip](wiki/languages/hip-cpp.md) — HIP C++ and the `__builtin_amdgcn_*` surface
- [lang-triton-rocm](wiki/languages/triton-rocm.md) — Triton on ROCm and its AMD-only launch parameters
- [lang-amdgcn-asm](wiki/languages/amdgcn-asm.md) — reading the ISA dump, and inline asm
- [lang-composable-kernel](wiki/languages/composable-kernel.md) — CK / CK-Tile

### Migration Guides

- [migration-cuda-to-hip](wiki/migration/cuda-to-hip.md) — CUDA (sm90) → HIP (gfx942)
- [migration-cdna-to-rdna4](wiki/migration/cdna-to-rdna4.md) — CDNA (gfx942) → RDNA4 (gfx1201)

## Source Repositories

PR coverage exists on the NVIDIA lane only; the AMD lane currently has docs,
blogs, and wiki pages. See `CLAUDE.md` for the AMD PR ingestion recipe.

| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | CUTLASS 4.x Blackwell support |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | SGLang Blackwell integration |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | vLLM Blackwell support |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | FlashInfer Blackwell kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | PyTorch/Inductor Blackwell |

## Competitions

- [GPU Mode NVFP4 Hackathon](sources/contests/gpu-mode-nvfp4/) — 4 NVFP4 kernel challenges on B200
- [FlashInfer MLSys 2026](sources/contests/flashinfer-mlsys26/) — MoE, Sparse Attention, GatedDeltaNet
