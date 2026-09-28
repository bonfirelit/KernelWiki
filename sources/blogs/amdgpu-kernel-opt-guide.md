---
id: blog-amdgpu-kernel-opt-guide
title: "AMDGPU Kernel Optimization Guide (nod-ai / amd-shark-ai)"
author: "nod-ai community"
url: https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md
source_category: community-note
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [mfma, lds, occupancy-tuning, instruction-scheduling, profiling]
retrieved_at: 2026-09-28
---

# AMDGPU Kernel Optimization Guide (nod-ai)

Community optimization guide for CDNA3/CDNA4 kernels: MFMA pipelining
patterns, LDS swizzle derivations, occupancy math, and an empirical LDS
bank/phase classifier harness. The per-architecture `ds_read` phase-group
tables (gfx942: 2×32/4×16/8×8 lane groups for b32/b64/b128; gfx950:
1×64/2×32/4×16) originate from this harness.

Caveat recorded by `blog-rocm-kernel-wiki`'s silicon pass: the harness's
b32/b64 phase groups reproduced on MI355X, but its b128 four-group
classification did not (inconclusive on an idle machine, three runs) — treat
the b128 lane-group tables as upstream-empirical.
