---
id: pattern-compute-bound
title: "Not Reaching Peak FLOPS"
type: pattern
tags: [tcgen05, 2sm-cooperative, pipeline-stages, warp-specialization]
symptoms: [compute-bound, low-tensor-core-utilization, pipeline-stalls]
candidate_techniques: [hw-2sm-cooperative, technique-pipeline-stages, technique-warp-specialization, technique-epilogue-fusion, technique-software-exp, technique-instruction-scheduling-amd, hw-mfma-cdna]
related: [pattern-low-sm-utilization, pattern-register-pressure]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, blog-flash-attention-4, blog-rocm-fp8-gemm-cdna4, blog-rocm-mxfp4-rotation]
---

## Symptom

Tensor-core work is the limiting path, measured memory bandwidth is not saturated, and achieved throughput is materially below the appropriate roofline. Do not use a universal utilization cutoff; compare the kernel with a shape- and datatype-matched bound.

## Likely Causes

1. **Pipeline bubbles**: MMA stalled waiting for data from TMA
2. **Non-matmul overhead**: Softmax, activation functions, reductions consuming cycles
3. **Insufficient arithmetic intensity**: the selected tile/decomposition does not keep the tensor pipeline fed
4. **Serialized epilogue work**: output conversion or stores delay the next mainloop iteration

## Candidate Techniques

| Technique | Effect |
|---|---|
| [2-SM cooperative](../hardware/2sm-cooperative.md) | Let a paired CTA group issue a supported collective MMA and share operand traffic |
| [Pipeline stages](../techniques/pipeline-stages.md) | Overlap TMA load with MMA compute |
| [Warp specialization](../techniques/warp-specialization.md) | Assign issuing and epilogue roles so independent phases can overlap |
| [Epilogue fusion](../techniques/epilogue-fusion.md) | Overlap epilogue with next tile's MMA |
| [Software exponential](../techniques/software-exp.md) | Distribute non-matmul ops across FMA units (FA4) |

## Example: FlashAttention-4

The FlashAttention-4 article reports that Blackwell's tensor throughput grew
faster than its special-function throughput for the comparison it presents.
Its kernel therefore evaluates a software exponential using range reduction and
a polynomial on general arithmetic units. That is a source-reported design for
the stated attention workload, not a general instruction replacement.

## Caveats
- 2-SM cooperative requires a compatible cluster launch and operand layout
- Pipeline depth is workload- and resource-dependent
- Software-emulated transcendentals trade accuracy for throughput

## On AMD (CDNA / RDNA4)

Two AMD-specific causes to rule out before reaching for scheduling.

**Wrong matrix-core shape.** Larger is not better. `doc-rocm-workload-optimization`
reports `mfma_16x16` typically beating `mfma_32x32` for GEMM on MI300X even at
large sizes, and `blog-rocm-mxfp4-rotation` picks `v_mfma_f32_16x16x32_bf16` over
32x32x16 because at M=1 the larger tile leaves the matrix core underutilized and
admits fewer concurrent waves. On gfx1201 this cause does not exist — WMMA is
16x16x16 only (`hw-wmma-rdna4`).

**Lock-stepped issue.** `blog-rocm-memory-scheduling` names the mechanism: waves
in a matmul kernel issue memory operations in sync, contend for the same
resources, and leave the matrix core waiting. The remedy is explicit
interleaving — `technique-instruction-scheduling-amd`.

Ordering matters here. In `kernel-cdna4-fp8-gemm`'s measured ladder, scheduling
was the *last* rung and worth 1.2x, after vectorization (11.2x), double buffering
(2.3x), and tiling. Reaching for `sched_barrier` before the loads vectorize is a
ten-fold misallocation of effort.
