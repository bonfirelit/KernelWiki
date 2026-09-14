---
id: hw-mfma-cdna
title: "MFMA on CDNA3 and CDNA4"
type: hardware
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [mfma, agpr, wave64, block-scale, mxfp4, mxfp8, ocp-fp8, ds-transpose]
confidence: source-reported
related: [hw-wmma-rdna4, hw-amd-narrow-precision, hw-lds, kernel-cdna4-fp8-gemm, migration-cdna-to-rdna4]
sources: [blog-salykova-matrix-cores-cdna, blog-rocm-fp8-gemm-cdna4, blog-rocm-occupancy-mi355x, blog-rocm-mxfp4-rotation]
aliases: [MFMA, "matrix core", v_mfma, "Matrix Fused Multiply Add"]
---

# MFMA on CDNA3 and CDNA4

CDNA's matrix core, `D := A*B + C`, issued by a **wave64** wavefront over a
per-instruction MxNxK shape. Unlike RDNA4's single 16x16x16 WMMA tile, MFMA is a
family, and picking the member is part of the optimization.

## Shape family

`blog-salykova-matrix-cores-cdna`, `Type (C,D) <- (A,B)`:

| Type | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|
| FP64<-FP64 | `16x16x4` | `16x16x4` |
| FP32<-FP32 | `32x32x2`, `16x16x4` | same |
| FP32<-FP16/BF16 | `32x32x8`, `16x16x16` | **plus** `16x16x32`, `32x32x16` |
| FP32<-FP8 | `16x16x32`, `32x32x16` | same |
| FP32<-FP8/FP6/FP4 | — | `16x16x128`, `32x32x64` |
| FP32<-MXFP8/MXFP6/MXFP4 | — | `16x16x128`, `32x32x64` |

## Programming contract

```cpp
// gfx942, wave64. 16x16x16 FP16: 4 A elements, 4 B elements, 4 C elements
// per lane (M*K/64, K*N/64, M*N/64).
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float    float4_t __attribute__((ext_vector_type(4)));

__global__ void mfma_16x16x16(const _Float16* A, const _Float16* B, float* D) {
  const int lane = threadIdx.x;                     // 0..63
  half4 a, b;
  float4_t c = {0.f, 0.f, 0.f, 0.f};

  for (int j = 0; j < 4; ++j) {
    a[j] = A[4 * (lane / 16) + 16 * (lane % 16) + j];
    b[j] = B[4 * (lane / 16) + 16 * (lane % 16) + j];
  }
  // cbsz, abid, blgp = 0
  c = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);

  for (int j = 0; j < 4; ++j)
    D[16 * (4 * (lane / 16) + j) + (lane % 16)] = c[j];
}
```

Per-lane element counts follow directly from wave64: `M*K/64` for A, `K*N/64`
for B, `M*N/64` for C. The FP8 `32x32x16` form is 8/8/16 and requires the first
two operands cast to `long`.

Builtin naming, verbatim from `blog-salykova-matrix-cores-cdna`:
`__builtin_amdgcn_mfma_f32_32x32x2f32`, `__builtin_amdgcn_mfma_f32_16x16x16f16`,
`__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8`,
`__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8`, and the gfx950-only
`__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`.

## Accumulators live in AGPRs

CDNA has a second register file. A well-scheduled GEMM keeps the whole C tile
resident there for the life of the loop: `blog-rocm-memory-scheduling` reports
`.agpr_count: 256` with zero spills, which is what removes C load/store traffic
from the inner loop entirely. `blog-rocm-occupancy-mi355x` states the rule
plainly — "The accumulator belongs in registers ... never in LDS" — and notes
that on CDNA4 VGPRs and AGPRs share one 512-register-per-lane file with neither
class exceeding 256.

## CDNA4 additions

- **Block-scaled MFMA**: `v_mfma_scale_f32_16x16x128_f8f6f4` and
  `v_mfma_scale_f32_32x32x64_f8f6f4` fold per-block microscaling into the matrix
  op. Scales are `uint8_t` E8M0, applied as `2^(scale - 127)` after the dot
  product and before accumulation; 127 means no scaling. See
  `hw-amd-narrow-precision`.
- **`ds_read_tr16_b64`** reads and transposes a 16-element LDS tile in one
  operation, feeding MFMA the right layout without a separate transpose pass
  (`blog-rocm-mxfp4-rotation`).
- **FP8 encoding changed**: CDNA3 uses FNUZ (E4M3FNUZ/E5M2FNUZ), CDNA4 uses OCP
  (E4M3FN/E5M2). Code that hard-codes one encoding silently changes numerics
  across the generations.

## Choosing the shape

Bigger is not better. `doc-rocm-workload-optimization` states that on MI300X
"`mfma_16x16` typically outperforms `mfma_32x32`, even for large tile/GEMM
sizes". `blog-rocm-mxfp4-rotation` picks `v_mfma_f32_16x16x32_bf16` over the
32x32x16 form because at M=1 the larger tile leaves the matrix core
underutilized and admits fewer concurrent waves.

## Transfers to RDNA4?

**No.** gfx1201 has no MFMA instruction, no AGPR file, and no `ds_read_tr`.
The transferable ideas are the *shape-selection* reasoning and the
accumulator-stays-in-registers rule; the instruction selection does not port.
See `migration-cdna-to-rdna4`.
