---
id: hw-amd-narrow-precision
title: "OCP FP8, MXFP8/MXFP6/MXFP4, and E8M0 block scales on AMD"
type: hardware
architectures: [gfx950, gfx942, gfx1201, cdna4, cdna3, rdna4]
tags: [ocp-fp8, mxfp4, mxfp8, block-scale, mfma, wmma, fine-grained-quantization]
confidence: source-reported
related: [hw-mfma-cdna, hw-wmma-rdna4, hw-nvfp4, technique-fine-grained-quantization, kernel-rdna4-wmma-gemm, kernel-cdna4-fp8-gemm]
sources: [blog-salykova-matrix-cores-cdna, blog-rocm-fp8-gemm-cdna4, blog-rocm-occupancy-mi355x, blog-rocm-mxfp4-rotation, blog-rdna4-wmma-lane-mapping, doc-amd-rdna4-matrix-cores, blog-rocm-kernel-wiki, doc-cdna4-isa]
aliases: ["OCP FP8", MXFP4, MXFP8, E8M0, "microscaling on AMD", FNUZ]
---

# OCP FP8, MXFP8/MXFP6/MXFP4, and E8M0 block scales on AMD

AMD's narrow-precision story is the OCP microscaling formats, and unlike
NVIDIA's NVFP4 (`hw-nvfp4`, UE4M3 scales over 16-element blocks) the scale type
here is **E8M0** over OCP MX blocks.

## E8M0 block scales

E8M0 is not an element type; it is a scale factor. Range `2^-127 .. 2^127`,
with `E=255` reserved for NaN. Scales are passed to the block-scaled MFMA
builtins as `uint8_t`, and the applied value is `2^(scale - 127)` — so **127
means no scaling**, not zero. The scale is applied after the dot product and
before accumulation (`blog-salykova-matrix-cores-cdna`).

## The FNUZ/OCP break between CDNA3 and CDNA4

This is the trap that produces wrong numbers rather than a compile error.
CDNA3 (gfx942) FP8 is **FNUZ**: E4M3FNUZ / E5M2FNUZ. CDNA4 (gfx950) FP8 is
**OCP**: E4M3FN / E5M2. A kernel that hard-codes one encoding — or a checkpoint
quantized against one — silently changes numerics when moved across the two
generations (`blog-salykova-matrix-cores-cdna`). The gap is concrete: max
finite E4M3 is **240** under FNUZ versus **448** under OCP, and the same byte
decodes to different reals under the two interpretations (measured on-device,
`blog-rocm-kernel-wiki`). A tensor ranged for 240 wastes half of OCP's
magnitude reach; one ranged for 448 has nowhere to go on FNUZ — there is no
Inf to land in.

## Hardware support by target

| Target | Native matrix-core narrow types |
|---|---|
| gfx942 (CDNA3) | FP8 (FNUZ) via `16x16x32`, `32x32x16` |
| gfx950 (CDNA4) | FP8 (OCP), FP6, FP4, and block-scaled MXFP8/MXFP6/MXFP4 via `16x16x128`, `32x32x64` |
| gfx1201 (RDNA4) | f16, bf16, fp8, int8, iu4 through WMMA. **No native FP4 matrix input.** |

On CDNA4 the scaled instructions are `v_mfma_scale_f32_16x16x128_f8f6f4` and
`v_mfma_scale_f32_32x32x64_f8f6f4`; A and B may use different types
independently. `blog-rocm-occupancy-mi355x` reports MI355X peaks of ~5 PFLOP/s
MXFP8 and 10 PFLOP/s MXFP6/FP4 dense.

## One E8M0 per 32 elements of K

The OCP MX spec fixes the block at **32 elements along K**, and the scaled
MFMA's K width is a multiple of it: one `v_mfma_scale_f32_16x16x128_f8f6f4`
consumes **4 scale blocks per operand** per instruction, the `32x32x64` form
**2** (`doc-cdna4-isa`). Scales are therefore a separate, small tensor — one
byte per 32 elements, ≈3% of the element bytes at FP8 and twice that share at
FP4 (`blog-rocm-kernel-wiki`, source-reported) — that must be staged so the
MMA never waits on it. Keep K tiles a multiple of 32 so each block maps to
exactly one scale byte; partial blocks need scalar fix-up.

For the `f8f6f4` ops the legacy `CBSZ`/`BLGP` modifier fields are repurposed
as **per-matrix format selectors** — CBSZ picks A's format, BLGP picks B's —
and mixed A/B formats are legal (`doc-cdna4-isa`, `blog-rocm-kernel-wiki`):

| Code | Format | Notes |
|---|---|---|
| `000` | E4M3 | OCP FP8 |
| `001` | E5M2 | OCP BF8 |
| `010` | E2M3 | FP6 |
| `011` | E3M2 | BF6 |
| `100` | E2M1 | FP4 |

`ABID[0]` gates whether the scale operands apply at all: with it clear, every
scale is forced to 1.0 and the scaled opcode degenerates to the unscaled
behavior.

```cpp
// gfx950 only. ROCm 7.2.4 clang-22 exposes ONLY the scaled f8f6f4 builtins
// -- there is no unscaled __builtin_amdgcn_mfma_f32_16x16x128_f8f6f4 on this
// toolchain (check your clang version). The scaled form takes 9 arguments:
// a(v8i32), b(v8i32), c(v4f32), then cbsz, blgp, opselA, scaleA, opselB,
// scaleB. The selectors must be immediates; the two E8M0 scales are runtime
// i32, with 127 the identity.
#include <hip/hip_runtime.h>

typedef int   i32v8 __attribute__((ext_vector_type(8)));
typedef float f32v4 __attribute__((ext_vector_type(4)));

__global__ void mxfp8_mma(const int* A, const int* B, float* D,
                          int scaleA, int scaleB) {
  const int lane = threadIdx.x;                  // 0..63
  i32v8 a, b;                                    // 32 B = a K=128 FP8 slice
  f32v4 c = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < 8; ++i) {
    a[i] = A[8 * lane + i];
    b[i] = B[8 * lane + i];
  }
  // cbsz=0, blgp=0 -> E4M3 for both A and B.
  c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
      a, b, c, /*cbsz=*/0, /*blgp=*/0, /*opselA=*/0, scaleA,
      /*opselB=*/0, scaleB);
  for (int j = 0; j < 4; ++j) D[4 * lane + j] = c[j];
}
```

Execution rate is dtype-dependent (source-reported, `blog-rocm-kernel-wiki`):
the `f8f6f4` op occupies the matrix pipe for **~32 cycles when either input
is FP8** and **~16 cycles for FP6/FP4**. That 2x is per SIMD — with all four
SIMDs of a CU co-issuing, it is exactly what doubles MI355X's MXFP6/MXFP4
peak over OCP-FP8; do not budget the factor a second time per CU.

The conversion side has hardware too: scale-aware pack converters, including
stochastic-rounding variants for quantization-aware training —
`v_cvt_scalef32_sr_pk_fp4_f32` and family (`doc-cdna4-isa`). The fused
epilogue below uses the dense sibling, `v_cvt_scalef32_pk_fp4_f32`.

## Doing MXFP4 without an FP4 matrix core

On gfx1201 the working pattern is dequantize-then-WMMA, and it is fast enough
to be worth it. `blog-rdna4-wmma-lane-mapping` loads 4-bit E2M1 weights with
E8M0 block scales, decodes them through an LDS-resident lookup table (`ldexpf`
applies the E8M0 exponent), and issues FP16 `16x16x16` WMMA with an FP32
accumulator — `TILE_K = 32` to match the E8M0 block size. See
`kernel-rdna4-wmma-gemm`.

```cpp
// E8M0 -> multiplier. 127 is the identity, not zero.
__device__ inline float e8m0_scale(unsigned char s) {
  return (s == 0xFF) ? __builtin_nanf("") : ldexpf(1.0f, (int)s - 127);
}

// One MXFP4 block: 32 packed E2M1 nibbles sharing one E8M0 exponent.
__device__ inline void dequant_mxfp4_block(const unsigned char* packed,
                                           unsigned char scale,
                                           const float* lut16,
                                           _Float16* out32) {
  const float s = e8m0_scale(scale);
  for (int i = 0; i < 16; ++i) {
    out32[2 * i + 0] = (_Float16)(lut16[packed[i] & 0xF] * s);
    out32[2 * i + 1] = (_Float16)(lut16[packed[i] >> 4]  * s);
  }
}
```

## Staying in registers across the quantization boundary

The fusion win is avoiding a global-memory round trip at the precision change.
`blog-rocm-mxfp4-rotation` feeds FP32 MFMA accumulators straight into
`v_cvt_scalef32_pk_fp4_f32` "without any register spill", computing the
per-group scale in a wave reduction first so everything stays in VGPRs. Measured
4.9x against the separated path on the MoE shape at M=1.
