---
id: blog-rocm-occupancy-mi355x
title: "Occupancy Math on the AMD MI355X GPU (CDNA4): A From-First-Principles Guide"
author: "Shekhar Pandey, Liz Li, Andy Luo (ROCm Blogs)"
url: https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
source_category: benchmark-blog
architectures: [gfx950, cdna4]
tags: [occupancy-tuning, mfma, lds, agpr, mxfp8, xcd]
retrieved_at: 2026-09-14
---

# Occupancy Math on the AMD MI355X GPU (CDNA4)

Published 2026-07-07. The measured source behind this wiki's claim that
occupancy is the wrong optimization target for MFMA-bound AMD kernels.

## Source-backed topics

- **Occupancy is the minimum of four limiters**, each expressed per SIMD or per
  CU and then clamped to the hardware cap of 8 waves per SIMD / 32 per CU:
  `floor(512 / VGPRs-per-lane)`, `floor(~800 / SGPRs-per-wave)`,
  `floor(160 KB / LDS-per-workgroup)`, and the resident-workgroup cap.
- **Allocation granularity**: "VGPRs round up to groups of 8", SGPRs to 16.
  (Note the difference from the MI300X figure of 16 in
  `doc-rocm-workload-optimization` — granularity is per-generation.)
- **MI355X numbers**: 160 KB LDS per CU shared across four SIMDs, up from 64 KB
  on CDNA3; 512 registers per lane with VGPR and AGPR sharing that file and
  neither exceeding 256; ~800 SGPRs per SIMD; 256 CUs as 8 XCDs x 32 CUs; up to
  2.4 GHz; 288 GB HBM3E at 8 TB/s; 256 MB Infinity Cache. Peak dense matrix
  throughput "~5 PFLOP/s of MXFP8 and 10 PFLOP/s of MXFP6/FP4 dense ... with
  structured sparsity pushing FP4 past 20 PFLOP/s".
- **Named scaled-MFMA instructions**: `v_mfma_scale_f32_16x16x128_f8f6f4`,
  `v_mfma_scale_f32_32x32x64_f8f6f4`, and `v_mfma_f32_16x16x128_f8f6f4` in the
  microbenchmark.
- **The measured counterintuitive result**: with ILP=8 the matrix core held
  4.82-4.84 PFLOP/s — "~97% of the MI355X GPU's ~5 PFLOP/s MXFP8 matrix peak" —
  all the way down to ~12% occupancy. At that 12% floor `MfmaUtil` read "~70%
  for ILP=1 but ~98% for ILP=8". A Gluon variant reached 4.97 PFLOP/s, "99% of
  the MXFP8 matrix peak", at full occupancy. The blog's conclusion: for
  MFMA-bound kernels "the sweet spot is routinely 2-4 waves/SIMD, not 8."
- **Accumulator placement**: "The accumulator belongs in registers — regular or
  accumulator VGPRs, the compiler's choice — never in LDS." The enlarged LDS is
  "a latency-hiding budget, not an accumulator substitute": spend it on deeper
  prefetch of the operand stream.
