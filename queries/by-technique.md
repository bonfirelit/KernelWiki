# Query: By Technique

> Auto-generated. Do not edit manually.

| Technique | Tags | Architectures | Confidence | Reproducibility | Sources |
|-----------|------|--------------|------------|-----------------|---------|
| [Branchless Bounds Checking with Buffer Descriptors](../wiki/techniques/buffer-oob-guard.md) | bounds-checking, vectorized-loads, gemm, attention | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 5 |
| [CCCL CUB SM100 Scan Tuning](../wiki/techniques/cccl-memory-primitives.md) | cuda-cpp, parallel-scan, vectorized-loads, tile-scheduling | sm100 | source-reported | snippet | 1 |
| [Choosing and Tuning the SDPA / FlashAttention Backend on ROCm](../wiki/techniques/rocm-attention-backends.md) | attention, flash-attention, triton-rocm, composable-kernel | gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4 | source-reported | snippet | 3 |
| [Chunk-Based Parallelism for Linear Recurrent Models](../wiki/techniques/chunk-parallelism.md) | chunk-parallelism, linear-attention, triton | sm90 | source-reported | snippet | 2 |
| [Double/Multi-Buffering Patterns](../wiki/techniques/double-buffering.md) | double-buffering, tmem, pipeline-stages | sm100, sm90 | source-reported | snippet | 3 |
| [Epilogue fusion](../wiki/techniques/epilogue-fusion.md) | epilogue-fusion, tmem, warp-specialization | sm100, sm90 | source-reported | snippet | 2 |
| [External Source-Map Research For Kernel Edits](../wiki/techniques/external-source-map-research.md) | cuda-cpp, cute-dsl, tma, wgmma | sm100, sm90 | source-reported | snippet | 5 |
| [Fine-grained FP8/FP4 scaling](../wiki/techniques/fine-grained-quantization.md) | fine-grained-quantization, fp8, fp4, nvfp4 | sm100, sm90 | source-reported | snippet | 3 |
| [Hand-Scheduling AMD Kernels (sched_barrier, sched_group_barrier, s_setprio)](../wiki/techniques/instruction-scheduling-amd.md) | instruction-scheduling, sched-barrier, mfma, lds | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 3 |
| [In-Register Transpose for WMMA Operands](../wiki/techniques/in-register-transpose.md) | in-register-transpose, wmma, ds-transpose, lds | gfx1201, gfx1200, rdna4, gfx950, cdna4 | source-reported | snippet | 3 |
| [Kernel fusion](../wiki/techniques/kernel-fusion.md) | kernel-fusion, fused-kernel, tmem | sm100, sm90 | source-reported | snippet | 4 |
| [LDS Bank Conflict Avoidance](../wiki/techniques/lds-bank-conflict-avoidance.md) | lds-bank-conflict-avoidance, lds, swizzling, shared-memory-optimization | gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4 | source-reported | snippet | 4 |
| [MFMA Software Pipelining — Keeping the Matrix Core Fed](../wiki/techniques/mfma-pipelining.md) | mfma-pipelining, mfma, sched-barrier, double-buffering | gfx942, gfx950, cdna3, cdna4 | source-reported | snippet | 5 |
| [Occupancy Tuning on AMD (and when not to)](../wiki/techniques/occupancy-tuning-amd.md) | occupancy-tuning, lds, agpr, mfma | gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4 | source-reported | snippet | 3 |
| [PTX Cache Policy Differentiation](../wiki/techniques/cache-policy.md) | cache-policy, vectorized-loads | sm100, sm90 | source-reported | snippet | 4 |
| [Persistent Kernels with Cluster Launch Control](../wiki/techniques/persistent-kernels.md) | persistent-kernel, clc, tile-scheduling | sm100 | source-reported | snippet | 4 |
| [Ping-Pong Scheduling](../wiki/techniques/ping-pong-scheduling.md) | ping-pong-scheduling, warp-specialization, tmem, pipeline-stages | sm100 | source-reported | snippet | 2 |
| [Preshuffled Weight Layouts for MFMA](../wiki/techniques/preshuffle-layout.md) | preshuffle-layout, mfma, data-reuse, vectorized-loads | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 2 |
| [Profiling Workflow on ROCm: counter to diagnosis](../wiki/techniques/profiling-workflow.md) | profiling, occupancy-tuning, lds, mfma | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 4 |
| [Register budgeting](../wiki/techniques/register-budgeting.md) | register-budgeting, register-reuse | sm100, sm90 | source-reported | snippet | 3 |
| [Shared Memory Swizzling](../wiki/techniques/swizzling.md) | swizzling, shared-memory-optimization, tma | sm100, sm90 | source-reported | snippet | 3 |
| [Software Pipelining and Multi-Stage Buffering](../wiki/techniques/pipeline-stages.md) | pipeline-stages, double-buffering, tma, mbarrier | sm100, sm90 | source-reported | snippet | 4 |
| [Software-Emulated Exponential](../wiki/techniques/software-exp.md) | software-exp, attention | sm100 | source-reported | snippet | 2 |
| [Split-K (GlobalSplitU) — Parallelizing the K Reduction](../wiki/techniques/split-k.md) | split-k, tile-scheduling, mfma, occupancy-tuning | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 1 |
| [Stream-K — Flat MAC-Iteration Decomposition over a Persistent Grid](../wiki/techniques/stream-k.md) | stream-k, persistent-kernel, tile-scheduling, split-k | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 1 |
| [Tile Scheduling Strategies](../wiki/techniques/tile-scheduling.md) | tile-scheduling, clc, persistent-kernel | sm100, sm90 | source-reported | snippet | 3 |
| [Warp Specialization on Hopper and Blackwell](../wiki/techniques/warp-specialization.md) | warp-specialization, tcgen05, tmem | sm100, sm90 | source-reported | snippet | 4 |
| [Wave-Level Reduction (DPP rows + ds_bpermute + readfirstlane)](../wiki/techniques/wave-reduction.md) | wave-reduction, cross-lane, reduction, wave64 | gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4 | source-reported | snippet | 5 |
| [Wide Vectorized Loads and Cache Policies](../wiki/techniques/vectorized-loads.md) | vectorized-loads, cache-policy, register-budgeting | sm100, sm90 | source-reported | snippet | 4 |
