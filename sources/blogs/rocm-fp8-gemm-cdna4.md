---
id: blog-rocm-fp8-gemm-cdna4
title: "FP8 GEMM Optimization on AMD CDNA4 Architecture"
author: "Jiahui Cao, Amanzhol Salykov, Andy Luo (ROCm Blogs)"
url: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
source_category: benchmark-blog
architectures: [gfx950, cdna4]
tags: [gemm, mfma, ocp-fp8, lds, lds-bank-conflict-avoidance, swizzling, double-buffering, global-load-lds, instruction-scheduling, ping-pong-scheduling]
retrieved_at: 2026-09-14
---

# FP8 GEMM Optimization on AMD CDNA4 Architecture

Published 2026-03-10. A full optimization ladder on one shape, each rung
measured — the closest AMD-side analogue of this wiki's DeepGEMM page.

## Source-backed topics

- **Measured ladder**, MI355X (gfx950) under ROCm 7.1.0, FP8 E4M3FN inputs,
  BF16 output, FP32 accumulation, M=N=K=4096, in TFLOP/s: naive 1.15; LDS
  tiling 4.80; matrix-core baseline 30.05; vectorized load 336.88; direct
  global-to-LDS 506.70; LDS swizzle 497.43; double buffering 1166.41;
  multi-wave 256x256_t512 2288.16; 8-wave ping-pong 2680.33; hipBLASLt 2750.42.
  At M=N=K=8192 the 8-wave ping-pong kernel reaches 3204.15 against hipBLASLt's
  3130.21.
- **Instruction mix**: the `16x16x128` FP8 MFMA form with FP32 accumulation;
  output tiles of 128x128 and 256x256 with a K-tile of 128. CDNA4's block-scaled
  additions are named as `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and
  `V_MFMA_SCALE_F32_32X32X64_F8F6F4`.
- **Scheduling intrinsics actually used**: `__builtin_amdgcn_sched_barrier(0)`
  ("no instructions may be scheduled across `sched_barrier`"),
  `__builtin_amdgcn_s_setprio(1)` / `__builtin_amdgcn_s_setprio(0)` (priority
  values 0-3), and `__builtin_amdgcn_s_barrier()`.
- **LDS swizzle**: `ds_read_b128` is performed in four phases and each phase
  must be conflict-free. The fix is "a row-based XOR remap on 16-byte columns",
  `swizzled_col = c ^ mask(r)` with `mask(r) = perm << 4`, which is its own
  inverse because XOR is self-inverse.
- **Direct global-to-LDS**: reached through `llvm_amdgcn_raw_buffer_load_lds`
  (`llvm.amdgcn.raw.buffer.load.lds`). CDNA4 widens the `GLOBAL_LOAD_LDS`
  per-lane transfer to "Up to 128 bits/lane", against 32-bit on CDNA3. Under
  double buffering "the next tile's global-to-LDS transfer runs in parallel with
  current MFMA work", synchronized with `s_waitcnt vmcnt(...)`.

## Boundary

The article does not use `sched_group_barrier` or `iglp_opt`, and it publishes
no scheduling-mask table. It does not discuss `ds_read_tr` or LDS padding.
