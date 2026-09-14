---
id: kernel-rdna4-wmma-gemm
title: "Fused MXFP4 -> FP16 WMMA GEMM on gfx1201"
type: kernel
architectures: [gfx1201, rdna4]
tags: [gemm, wmma, mxfp4, block-scale, lds, kernel-fusion, in-register-transpose]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, quantization]
languages: [hip]
related: [hw-wmma-rdna4, hw-gfx1201, hw-amd-narrow-precision, hw-lds, technique-in-register-transpose, technique-lds-bank-conflict-avoidance, migration-cdna-to-rdna4]
sources: [blog-rdna4-wmma-lane-mapping, doc-amd-rdna4-matrix-cores]
performance_claims:
  - gpu: "Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB)"
    dtype: "MXFP4 weights dequantized to FP16, FP32 accumulate"
    shape: "peak over the author's sweep; correctness verified up to 17408x5120. Exact maximizing shape not stated."
    metric: TFLOPS
    value: 40.8
    utilization: "53% of FP16 WMMA theoretical"
    source_id: blog-rdna4-wmma-lane-mapping
    source_locator: "README, 'fused MXFP4 GEMM kernel hitting 40.8 TFLOPS' and the 53%-of-theoretical note; ROCm 7.1.0, hipcc (clang-19)"
---

# Fused MXFP4 -> FP16 WMMA GEMM on gfx1201

The reference point for what a hand-written RDNA4 matrix kernel achieves, on the
same silicon as this host. It solves a problem that has no hardware answer on
gfx12: **there is no native FP4 WMMA input type**, so 4-bit weights must be
decoded before they reach the matrix core, and the whole question is whether the
decode can be hidden.

## Shape of the kernel

Per `blog-rdna4-wmma-lane-mapping`:

- **Pipeline**: load MXFP4 weights (4-bit E2M1 elements plus E8M0 block scales)
  -> dequantize through an LDS-resident lookup table -> FP16 `16x16x16` WMMA with
  an FP32 accumulator.
- **`TILE_K = 32`**, chosen to match the E8M0 block size — one block scale is
  consumed per K-tile, so the scale never has to be re-fetched mid-tile.
- 2x2 register tiling, 64x64 block, 4 warps, ~9 KB of LDS.
- `ldexpf` applies the E8M0 exponent during decode.

Measured 40.8 TFLOPS, 53% of FP16 WMMA theoretical, and 3.8x faster than
separate dequantization followed by a hipBLAS GEMM for batch size <= 32.

## The decode inner loop

```cpp
// One MXFP4 block -> 32 FP16 values, through an LDS LUT. The LUT is 16 floats:
// every possible E2M1 nibble. Decode cost per element is one LDS read and one
// multiply, which is cheap enough to sit inside the K-loop.
__device__ inline void mxfp4_block_to_half(const unsigned char* __restrict__ packed,
                                           unsigned char scale_e8m0,
                                           const float* __restrict__ lds_lut16,
                                           _Float16* __restrict__ out32) {
  const float s = ldexpf(1.0f, (int)scale_e8m0 - 127);   // 127 == no scaling
  for (int i = 0; i < 16; ++i) {
    const unsigned char byte = packed[i];
    out32[2 * i + 0] = (_Float16)(lds_lut16[byte & 0xF] * s);
    out32[2 * i + 1] = (_Float16)(lds_lut16[byte >> 4]  * s);
  }
}
```

## Why the fusion pays

The alternative is a dequantize kernel writing FP16 weights to HBM followed by a
GEMM reading them back. At batch size <= 32 the GEMM is memory-bound on the
weights, so that round trip dominates: the 3.8x is almost entirely the HBM
traffic that fusion removes, not arithmetic. The same reasoning drives the CDNA
result in `blog-rocm-mxfp4-rotation`, where keeping accumulators in VGPRs across
a precision change is worth 4.9x on the MoE shape at M=1.

## Performance boundary

53% of theoretical is the honest headline: this is a fused, quantized,
low-batch kernel, not a dense-GEMM peak number, and the remaining 47% is decode
and LDS traffic that a native FP4 matrix core would not pay. Do not compare it
against a CDNA4 dense FP8 figure such as the 2680 TFLOP/s in
`kernel-cdna4-fp8-gemm` — different precision, different silicon class,
different arithmetic intensity.

The measurement is a single author's sweep under ROCm 7.1.0; the exact
maximizing shape is not stated, and this wiki has not reproduced it. The lane
mapping the kernel depends on **has** been reproduced here — see `hw-wmma-rdna4`,
0 of 256 accumulator elements mismatched against a CPU reference on this host.
