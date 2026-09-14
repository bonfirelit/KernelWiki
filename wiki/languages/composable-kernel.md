---
id: lang-composable-kernel
title: "Composable Kernel (CK / CK-Tile)"
type: language
tags: [composable-kernel, mfma, gemm, attention, lds]
related: [lang-hip, lang-triton-rocm, hw-mfma-cdna, hw-lds, kernel-cdna4-fp8-gemm, technique-instruction-scheduling-amd]
sources: [doc-rocm-workload-optimization, blog-rocm-fp8-gemm-cdna4, blog-salykova-matrix-cores-cdna]
reproducibility: snippet
architectures: [gfx942, gfx950, cdna3, cdna4]
confidence: inferred
aliases: [CK, "CK-Tile", "Composable Kernel"]
---

# Composable Kernel (CK / CK-Tile)

AMD's templated device-library layer, and the structural counterpart of CUTLASS:
a GEMM or attention pipeline expressed as a composition of tile descriptors,
thread-to-data mappings, and a matrix-core pipeline policy, specialized at
compile time. It is the backend behind the CK path of FlashAttention on ROCm and
much of hipBLASLt's tuned surface.

## Where it sits

```cpp
// The shape of a CK-style device op instantiation: every layout, tile size, and
// pipeline decision is a template parameter resolved at compile time, exactly
// as in CUTLASS. This is illustrative structure, not a copy of a CK header.
using DeviceGemmFp8 = ck::tensor_operation::device::DeviceGemmXdl<
    ALayout, BLayout, CLayout,          // row/column major per operand
    ck::f8_t, ck::f8_t, ck::bhalf_t,    // A, B, C element types
    float,                              // accumulator
    PassThrough, PassThrough, PassThrough,
    /* BlockSize  = */ 256,
    /* MPerBlock  = */ 256,
    /* NPerBlock  = */ 256,
    /* KPerBlock  = */ 128,             // the K-tile; cf. the 128 in the FP8 ladder
    /* MPerXdl    = */ 16,              // matrix-core tile: 16 beats 32 for GEMM
    /* NPerXdl    = */ 16,
    /* NumPrefetch= */ 2>;              // double buffering

auto gemm = DeviceGemmFp8{};
auto argument = gemm.MakeArgument(a_ptr, b_ptr, c_ptr, M, N, K,
                                  stride_a, stride_b, stride_c,
                                  PassThrough{}, PassThrough{}, PassThrough{});
if (!gemm.IsSupportedArgument(argument))
  return;                               // shape/alignment rejected at runtime
gemm.GetInvoker().Run(argument, {stream});
```

The template parameters are not arbitrary. They are the same decisions this wiki
documents from first principles elsewhere: `MPerXdl`/`NPerXdl` is the matrix-core
shape choice from `hw-mfma-cdna` (16 over 32 for GEMM), `KPerBlock` is the
K-tile, `NumPrefetch` is `technique-double-buffering`, and the LDS layout policy
is `technique-lds-bank-conflict-avoidance`.

## Selection guidance

Use CK when a tuned instantiation for your shape and dtype already exists — it
will start where a hand-written kernel finishes. Write HIP directly
(`lang-hip`) when the fusion you need has no descriptor, or when the win is in
the *schedule* rather than the tiling: `blog-rocm-fp8-gemm-cdna4` reaches 2680
TFLOP/s against hipBLASLt's 2750 on M=N=K=4096 with a hand-scheduled HIP kernel,
and wins outright at 8192 (3204 vs 3130). A library that is 3% ahead on one
shape is not necessarily ahead on yours.

## Boundary

`confidence: inferred`. The CK-specific claims on this page are structural
(what the layer is for, how instantiation works, how its parameters map onto
techniques documented elsewhere here). No CK source document has been captured
into `sources/` yet, and the snippet above is illustrative of the API *shape*,
not quoted from a CK header — check names and parameter order against the
version of CK you build against. CK targets CDNA; there is no RDNA4 story on
this page.
