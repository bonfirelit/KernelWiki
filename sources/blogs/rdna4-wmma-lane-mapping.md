---
id: blog-rdna4-wmma-lane-mapping
title: "rdna4-wmma-guide — WMMA lane mapping for gfx12 and a fused MXFP4 GEMM"
author: JohnTDI-cpu
url: https://github.com/JohnTDI-cpu/rdna4-wmma-guide
source_category: benchmark-blog
architectures: [gfx1201, rdna4]
tags: [wmma, mxfp4, block-scale, lds, gemm]
retrieved_at: 2026-09-14
---

# rdna4-wmma-guide

Community derivation of the gfx12 WMMA accumulator lane mapping, plus a fused
MXFP4 GEMM measured on the same silicon family as this host. Licensed CC BY 4.0.

## Source-backed topics

- **Accumulator (D/C) fragment mapping** for `v_wmma_f32_16x16x16_f16` in
  wave32, stated as a "column-distributed fragment layout":

      VGPR[lane][j] = matrix[(lane / 16) * 8 + j][lane % 16]
      matrix[row][col] -> VGPR[(row / 8) * 16 + col][row % 8]

  In kernel terms `col = lane % 16`, `row_base = (lane / 16) * 8`,
  `row = row_base + j`. Lanes 0-15 cover columns 0-15 rows 0-7; lanes 16-31
  cover columns 0-15 rows 8-15. The lane index selects the **column** (N) and
  the accumulator register index selects the row (M).
- **Intrinsics exercised**:
  `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` and
  `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`.
- **MXFP4 is not a native WMMA input type here.** The kernel dequantizes 4-bit
  E2M1 values with E8M0 block scales into FP16 and then issues FP16 WMMA.
- **Kernel shape**: load MXFP4 weights -> LDS-based LUT dequant (with `ldexpf`
  for the E8M0 exponent) -> FP16 `16x16x16` WMMA with FP32 accumulator.
  `TILE_K = 32` to match the E8M0 block size, 2x2 register tiling, 64x64 block,
  4 warps, ~9 KB of LDS.
- **Measured**: 40.8 TFLOPS peak, "53% of FP16 WMMA theoretical", on an AMD
  Radeon AI PRO R9700 (gfx1201, RDNA4, 32 GB) under ROCm 7.1.0 with hipcc
  (clang-19). Reported 3.8x faster than separate dequant plus hipBLAS GEMM for
  batch size <= 32; correctness verified up to 17408x5120.

## Boundary

The guide documents the **output/accumulator** fragment only. It gives no
separate A or B input-fragment lane table, and it makes no SWMMAC or structured
sparsity claim.
