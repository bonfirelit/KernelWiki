---
id: blog-rocm-mxfp4-rotation
title: "Production-Ready MXFP4 Online Rotation with Fused Kernels on AMD Instinct MI355X"
author: "Jiangyong Ren, Felix Marty, Xinjun Niu, Chao Li, Lin Zhao, Wei Luo, Bowen Bao, Spandan Tiwari, Ashish Sirasao (ROCm Blogs)"
url: https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-fused-rotation/README.html
source_category: benchmark-blog
architectures: [gfx950, cdna4]
tags: [mxfp4, mfma, ds-transpose, lds, kernel-fusion, quantization]
retrieved_at: 2026-09-14
---

# MXFP4 Online Rotation with Fused Kernels on MI355X

Published 2026-08-13. The source for CDNA4's LDS transpose read and for tile
selection under an extremely skinny M.

## Source-backed topics

- **`ds_read_tr16_b64`** is "a CDNA4 hardware instruction that reads and
  transposes a 16-element tile from LDS in a single operation". It exists to
  hand MFMA a correctly laid-out operand without a separate transpose pass. The
  rotation matrix is tiled into LDS so each wave reads its tile once and reuses
  it across MFMA iterations.
- **Tile choice under low arithmetic intensity**: `v_mfma_f32_16x16x32_bf16`
  was chosen over the 32x32x16 form. At M=1 the rotation matmul is "severely
  compute-underutilized", so the smaller tile admits more concurrent waves. The
  larger tile also "ran into lane mapping complexity".
- **Staying in registers across a fusion boundary**: FP32 MFMA accumulators feed
  `v_cvt_scalef32_pk_fp4_f32` directly "without any register spill". The
  per-group scale is produced by a wave reduction beforehand, "keeping
  everything in VGPRs", which removes an intermediate BF16 global-memory round
  trip.
- **Measured, MI355X**: dense path (K=4096, rotation_size=128) 12.23 us
  separated vs ~6.3 us fused at M=1 (~2.0x), 13.45 vs ~6.5 us at M=32 (~2.1x).
  MoE path (K=2048, rotation_size=128, TOPK=8) 39.1 vs 8.0 us at M=1 (4.9x) and
  36.6 vs 9.4 us at M=256 (3.9x). Switching the rotation-matrix load to bf16x8
  vectorized form cut memory access time 22% at M=1.
