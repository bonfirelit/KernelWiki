---
id: blog-salykova-matrix-cores-cdna
title: "Matrix Core Programming on AMD CDNA3 and CDNA4 architecture"
author: "Amanzhol Salykov, Andy Luo, Carlus Huang, Peng Sun"
url: https://salykova.github.io/matrix-cores-cdna
source_category: community-note
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [mfma, mxfp4, mxfp8, ocp-fp8, block-scale, wave64]
retrieved_at: 2026-09-14
---

# Matrix Core Programming on AMD CDNA3 and CDNA4

Published 2025-09-30. The reference write-up for MFMA shapes, builtins, and
per-lane operand layouts on CDNA.

## Source-backed topics

- **Shape table** (`Type (C,D) <- (A,B)`, `MxNxK`), CDNA3 -> CDNA4:
  FP64<-FP64 `16x16x4` (both); FP32<-FP32 `32x32x2`, `16x16x4` (both);
  FP32<-FP16/BF16 `32x32x8`, `16x16x16` on CDNA3, **plus** `16x16x32`,
  `32x32x16` on CDNA4; FP32<-FP8 `16x16x32`, `32x32x16` (both);
  FP32<-FP8/FP6/FP4 and FP32<-MXFP8/MXFP6/MXFP4 `16x16x128`, `32x32x64`,
  **CDNA4 only**.
- **Builtin naming**, verbatim examples:
  `__builtin_amdgcn_mfma_f32_32x32x2f32`,
  `__builtin_amdgcn_mfma_f32_16x16x16f16`,
  `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8`,
  `__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8`, and the gfx950-only
  `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4`. General form:
  `d_reg = __builtin_amdgcn_mfma_ODType_MxNxKInDType(a_reg, b_reg, c_reg, cbsz, abid, blgp)`;
  the scaled form appends `Atype, Btype, OPSEL_A, scale_a, OPSEL_B, scale_b`.
- **Per-lane element counts** at wavefront size 64: `M*K/64` for A, `K*N/64` for
  B, `M*N/64` for C. Worked cases: `32x32x2` FP32 is 1/1/16; `16x16x16` FP16 is
  4/4/4, with A loaded as `fp16x4_t` at index `4*(threadIdx.x/16) + 16*(threadIdx.x%16)`;
  `32x32x16` FP8 is 8/8/16 and requires the first two operands cast to `long`;
  scaled `32x32x64` is 32 (A) / 1 (Ax) / 32 (B) / 1 (Bx) / 16 (C), with the
  scales "applied after the normal dot product and prior to output/accumulation."
- **E8M0 block scales**: the scale type for microscaling and block-scaled MFMA,
  range `2^-127 .. 2^127` with `E=255` reserved for NaN. Scales are passed as
  `uint8_t`; the applied value is `2^(scale - 127)`, so 127 means no scaling.
- **FP8 encoding differs by generation**: CDNA3 uses FNUZ (E4M3FNUZ / E5M2FNUZ);
  CDNA4 uses the OCP forms (E4M3FN / E5M2). Code that hard-codes one encoding
  silently changes numerics when moved between the two.
- FP4 helpers named: `__amd_extract_fp4`, `__amd_create_fp4x2`, and the
  `__amd_fp8_storage_t` / `__amd_fp4x2_storage_t` types from `hip_ext_ocp.h`.
  The FP4 builtin takes 256-bit operands, so `fp4x64_t` pads 128 data bits with
  128 zero bits.
