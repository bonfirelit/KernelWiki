---
id: technique-preshuffle-layout
title: "Preshuffled Weight Layouts for MFMA"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [preshuffle-layout, mfma, data-reuse, vectorized-loads]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma-cdna]
related: [technique-in-register-transpose, technique-fine-grained-quantization, pattern-memory-bound]
sources: [blog-rocm-kernel-wiki, doc-cdna3-isa]
symptoms: [memory-bound, low-compute-utilization]
---

# Preshuffled Weight Layouts

Every `v_mfma_*` consumes its A and B fragments in a fixed, instruction-shape-
specific lane/VGPR interleave (`hw-mfma-cdna`, `doc-cdna3-isa`). A weight tile
stored row-major is in the *wrong* order, so a naive kernel re-derives the
interleave at runtime: strided `ds_read` gathers, address math, and the
attendant LDS bank conflicts — repeated for every K-iteration of every tile
for the life of the kernel.

Preshuffling moves that permutation **off the critical path**. Weights are
static during inference, so permute them once — offline, or in a one-time
prologue at model load — into the exact order the MFMA lanes consume. The hot
loop degenerates to a flat vectorized copy: consecutive bytes in memory are
consecutive elements for one lane, then the next lane. No index math, no
swizzle, conflict-free by construction.

```cpp
#include <hip/hip_runtime.h>
using f16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using f32x4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

// Consumer side. The packing step laid B out so that lane l's four f16
// elements for v_mfma_f32_16x16x16_f16 are contiguous — one flat 64-bit
// load feeds the matrix core with zero runtime swizzle.
__device__ f32x4 mfma_k_step(const f16x4* __restrict__ a_shuf,  // [k][64]
                             const f16x4* __restrict__ b_shuf,  // [k][64]
                             f32x4 acc, int k) {
  const int lane = threadIdx.x & 63;              // MFMA is wave64
  const f16x4 a = a_shuf[k * 64 + lane];          // already in MFMA order
  const f16x4 b = b_shuf[k * 64 + lane];
  return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}
```

Derive the permutation from AMD's Matrix Instruction Calculator, or read it
out of a known-good kernel's disassembly, and bake it into the weight-packing
step — never hand-transcribe lane indices (`blog-rocm-kernel-wiki`).

## Where it pays, and where it does not

- **Inference GEMM/GEMV**: weights are frozen; shuffle once, amortize over
  every token. Highest payoff.
- **FP8**: the narrower the element, the uglier the interleave (32-element K
  slices packed across lanes), so the removed runtime swizzle is worth more.
  Pairs naturally with `technique-fine-grained-quantization` layouts.
- **Per-expert MoE weights**: static per expert; preshuffle each expert's
  tiles at load.
- **Activations: no.** They are produced fresh each run, so you would pay the
  full shuffle every time — that is just `technique-in-register-transpose`
  with extra steps. Preshuffle is strictly a static-operand optimization;
  `pattern-memory-bound` GEMV is where the removed gather traffic shows up
  first.

## Pitfalls

- The layout is **instruction-shape specific**: a buffer packed for
  `16x16x16` is wrong for `32x32x8`. Re-pack if you retune the MFMA shape.
- The layout is **architecture specific**: gfx942 vs gfx950 differ in FP8
  *encoding* (FNUZ vs OCP) even where the *permutation* matches
  (`blog-rocm-kernel-wiki`).
- Keep each lane's run 128-bit aligned; a correct permutation that breaks
  vector width quietly erases the gain.
- A wrong permutation computes garbage **without faulting**. Diff against a
  row-major reference before benchmarking anything.

## Transfers to RDNA4?

**Yes — and it is arguably worth more there.** gfx1201 has WMMA (wave32,
fixed 16x16x16) instead of MFMA, and its operand layout problem is the same
one, minus a relief valve: CDNA lets accumulators spill over into AGPRs while
VGPRs do the addressing dance, but RDNA4 has no AGPR file — every register the
runtime swizzle burns comes out of the single VGPR file that also holds WMMA
operands and accumulators. Moving the permutation to load time is exactly the
cost removal `technique-in-register-transpose` pays at runtime. Regenerate the
mapping for the WMMA `_gfx12` builtins; do not reuse a CDNA packing.
