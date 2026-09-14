---
id: doc-amd-rdna4-matrix-cores
title: "Using the Matrix Cores of AMD RDNA 4 architecture GPUs"
author: "Takahiro Harada and Atsushi Yoshimura (AMD GPUOpen)"
url: https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
source_category: official-doc
architectures: [gfx1201, gfx1200, rdna4, rdna3]
tags: [wmma, wave32]
retrieved_at: 2026-09-14
---

# Using the Matrix Cores of AMD RDNA 4 architecture GPUs

AMD's own introduction to WMMA on gfx12. Published 2025-07-11.

## Stable guidance used by this wiki

- Intrinsic naming convention:
  `__builtin_amdgcn_wmma_<C,D format>_16x16x16_<A,B format>_w32_gfx12`. The
  worked example is `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`. The
  `_gfx12` postfix "did not exist for the WMMA intrinsics for RDNA 3 or gfx11
  generation."
- The intrinsic must be called from every lane in the wavefront; a wavefront is
  32 lanes.
- WMMA "operates on matrices of 16x16 dimension only". Smaller problems are
  padded; larger ones are decomposed into 16x16 GEMMs.
- Each lane holds **8 elements** of a matrix (8 x 32 lanes = 256 = 16 x 16).
  The article's index helpers are `laneWrapped = threadIdx.x % 16` and
  `laneGroup = threadIdx.x / 16`, with `WMMA_DATA_WIDTH = 8`.
- Fragment orientation: "B, C, and D matrices are row major while A is
  transposed thus column major."
- Change from RDNA3: the VGPR layout "does not have backward compatibility".
  RDNA3 required duplicating some A and B elements — "it is removed for
  RDNA 4". RDNA3 also split C/D into even/odd halves across the lower and
  upper 16 lanes, forcing a lane shuffle when chaining WMMA operations; "This
  is not needed for RDNA 4."
- Chaining D into a following B operand still needs a downcast, because "the D
  matrix is 32-bit float while the C matrix is 16-bit float". The article uses
  `__builtin_amdgcn_cvt_pkrtz` to convert and pack two f32 into one VGPR.
- Stated theoretical throughput per CU per clock on RX 9070 XT: FP16 1024,
  BF16 1024, I8 2048.
