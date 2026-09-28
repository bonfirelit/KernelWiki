---
id: lang-flydsl
title: "FlyDSL"
type: language
tags: [flydsl, mfma, gemm, moe, attention]
related: [lang-hip, lang-composable-kernel, hw-mfma-cdna, technique-preshuffle-layout]
sources: [blog-flydsl-kernel-profiling, blog-rocm-kernel-wiki, doc-cdna4-isa]
reproducibility: snippet
architectures: [gfx942, gfx950, cdna3, cdna4]
confidence: experimental
aliases: [FlyDSL, "fly dialect"]
---

# FlyDSL

**Status: experimental and pre-production.** Dialect ops, pass names, and the
Python API move between commits — pin a commit and treat every snippet here as
a sketch of the programming model. For production AMD kernels today, use
`lang-composable-kernel` or `lang-triton-rocm`.

FlyDSL is a Python + MLIR DSL that describes AMD kernels as **layout algebra**
rather than pointer arithmetic: a CuTe-style `(Shape, Stride)` pair mapping
coordinates to linear offsets, carried in a custom `fly` MLIR dialect. A kernel
is traced from Python (`@flyc.kernel`), lowered Fly → ROCDL → LLVM IR, and
compiled to a HIP fatbin (`@flyc.jit`). It targets the matrix cores through
**MFMA atoms** — an instruction bundled with the fragment layouts for A, B,
and the accumulator — so it is naturally GEMM-shaped, and its verified targets
are gfx942/gfx950 (`hw-mfma-cdna`). In practice it shows up as an optional
AITER backend for MoE expert GEMMs, where `technique-preshuffle-layout`
packing is expressible as a layout transform instead of an index hack.

```python
from fly import Shape, Stride, Layout

# A 128x64 tile, column-major (leading dim 128): idx = sum(coord_i * stride_i)
A = Layout(Shape(128, 64), Stride(1, 128))
assert A.crd2idx((1, 0)) == 1       # down a column
assert A.crd2idx((0, 1)) == 128     # across a row
```

Because the layout is data, the compiler folds `crd2idx` chains at trace time
and can compose the global→LDS copy, the LDS→VGPR operand read, and the
pre-shuffled weight layout as three views of one algebra — the bookkeeping
that is error-prone by hand for MFMA's per-lane register distributions. When
results are wrong, debug at the ROCDL stage: dump the lowered IR and `grep`
for the intended `llvm.amdgcn.mfma.*` intrinsic before suspecting the algebra
(`doc-cdna4-isa` for what the instructions should be).

## Measured on MI350X: wins, parity, and honest headroom

The unusual asset here is a first-party profiling sweep
(`blog-flydsl-kernel-profiling`): 17 FlyDSL kernels on MI350X (gfx950),
captured with rocprofv3 ATT + hardware counters, against matched-shape
AITER / CK / hipBLASLt baselines. Ratios are FlyDSL throughput ÷ baseline.

| Bucket | Kernel | vs baseline |
|---|---|---|
| WIN | softmax | **2.05x** vs Triton |
| WIN | hgemm_splitk | **1.66x** vs CK / hipBLASLt |
| WIN | moe_gemm | **1.11x** vs AITER |
| PARITY | layernorm, quant | ~1.0x vs AITER |
| HEADROOM | flash_attn | 0.92x vs CK-tile |
| HEADROOM | paged-attention | 0.48x vs AITER |
| HEADROOM | rope | **0.17x** vs AITER |

Two findings matter beyond the table:

- **The dominant headroom root cause is VGPR pressure.** The attention and
  GEMM losers live at 1-2 resident waves/SIMD with 175-251 VGPRs live — the
  `pattern-register-pressure` signature. Cutting the live VGPR set far enough
  to admit a second wave is the lever, per the sweep's counter data; this is
  the same diagnosis `technique-occupancy-tuning-amd` reaches from the other
  direction.
- **Check the roofline before optimizing.** The softmax fast path was found
  dead-coded behind `False and` — and re-enabling it moved the kernel only
  ±7%, because both paths already saturate HBM (~5 TB/s) and buffer the whole
  row in registers; vectorization trimmed instruction count, not bytes. The
  2.05x headline is FlyDSL-vs-Triton, not fast-path-vs-scalar. Run the
  `technique-profiling-workflow` roofline check first and this surprise does
  not happen.

Also recorded honestly there: `fp8_gemm_4wave` fails to compile on the
fast-dispatch path — a real regression, not a tuning gap. Experimental means
experimental.

## Gotchas

- **wave64 targets.** gfx942/gfx950 are wave64-only; never reuse an atom or
  register layout across architectures without revalidation.
- **FP8 re-encodes between gfx942 and gfx950** (FNUZ vs OCP) — a layout built
  for one is not bit-compatible with the other.
- **Moving API.** The MMA-atom surface (`make_mma_atom` / `mma_atom_call_ssa`)
  recently replaced raw ROCDL intrinsic emission; expect further churn.

## Transfers to RDNA4?

The DSL advertises gfx1201 among its verified targets, so the layout algebra
itself is arch-neutral. But every number above is gfx950 silicon, and the
atom layer is MFMA-shaped — on RDNA4 the corresponding hardware is WMMA
(`hw-wmma-rdna4`), and there is no published sweep showing the DSL's codegen
quality there. Treat the profiling verdicts as CDNA-only evidence; if you run
FlyDSL on gfx1201, you are the experiment.
