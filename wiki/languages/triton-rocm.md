---
id: lang-triton-rocm
title: "Triton on ROCm"
type: language
tags: [triton-rocm, occupancy-tuning, mfma, lds]
related: [lang-triton, lang-hip, technique-occupancy-tuning-amd, hw-mfma-cdna, hw-lds, pattern-register-pressure]
sources: [doc-rocm-workload-optimization, blog-rocm-occupancy-mi355x]
reproducibility: snippet
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
confidence: source-reported
aliases: ["Triton on ROCm", "ROCm Triton"]
---

# Triton on ROCm

Triton's ROCm backend is the practical starting point for AMD kernels that are
shape-parametric: you write once and autotune, and the compiler picks MFMA
shapes and LDS layouts. The AMD-specific part is a small set of extra
`triton.jit` launch parameters that have no CUDA counterpart.

## AMD-specific launch parameters

```python
import triton
import triton.language as tl

@triton.jit
def gemm(a_ptr, b_ptr, c_ptr, M, N, K,
         stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (offs_k[None, :] + k * BLOCK_K) * stride_ak)
        b = tl.load(b_ptr + (offs_k[:, None] + k * BLOCK_K) * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)


gemm[(triton.cdiv(M, 128), triton.cdiv(N, 128))](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
    num_warps=4,
    num_stages=2,             # 2: single GEMM. 1: two fused GEMMs (FlashAttention).
    waves_per_eu=2,           # VGPR-pressure hint; only for VGPR-limited kernels.
    matrix_instr_nonkdim=16,  # 16 -> mfma_16x16, 32 -> mfma_32x32
)
```

Per `doc-rocm-workload-optimization`:

- **`num_stages`** — 2 for a single GEMM; **1** for two fused GEMMs, which is the
  FlashAttention shape; 2 for a GEMM fused with a non-GEMM operator; 1 when
  there is no GEMM. This inverts the CUDA-side instinct that more stages is
  better.
- **`matrix_instr_nonkdim`** — 16 selects `mfma_16x16`, 32 selects `mfma_32x32`.
  The documented default advice is 16: "For GEMM kernels on an MI300X GPU,
  `mfma_16x16` typically outperforms `mfma_32x32`, even for large tile/GEMM
  sizes."
- **`waves_per_eu`** — a VGPR-reduction hint to the LLVM backend. See
  `technique-occupancy-tuning-amd` for the narrow condition under which it helps,
  and for why maximizing occupancy is usually the wrong objective on CDNA.

## Inspecting what you got

```bash
AMDGCN_ENABLE_DUMP=1 python gemm.py 2>&1 | grep -E '\.vgpr_count|\.agpr_count'
MLIR_ENABLE_DUMP=1   python gemm.py 2>&1 | grep -E 'triton_gpu.shared|triton_gpu.num-warps'
TORCH_COMPILE_DEBUG=1 python model.py     # extract Inductor-generated kernels
```

Those three numbers — VGPR count, LDS bytes, waves per workgroup — are exactly
the inputs to the occupancy formula in `technique-occupancy-tuning-amd`. The
ISA checklist to apply to the dump is in `hw-amd-memory-ops`: confirm
`global_load_dwordx4` for device loads, `_b128` forms for LDS traffic, and read
the `s_waitcnt` operands to see whether the loop was pipelined.

## Selection guidance

Triton on ROCm gets a competent MFMA GEMM without hand-writing fragment
layouts, and autotuning over `BLOCK_*` plus the four parameters above recovers
most of the available performance for standard shapes. Drop to `lang-hip` when
you need a specific instruction the compiler will not select, a hand-placed
`s_waitcnt`, or the scheduling control in
`technique-instruction-scheduling-amd` — `blog-rocm-fp8-gemm-cdna4` shows the
top ~2.3x of its FP8 ladder living in exactly that territory.

## Boundary

`triton-rocm` is tracked as a language distinct from `lang-triton` here because
the launch-parameter surface and the tuning advice genuinely differ. Kernel
*logic* is portable between the two backends; the tuning is not.
