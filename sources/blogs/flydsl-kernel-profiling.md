---
id: blog-flydsl-kernel-profiling
title: "FlyDSL Kernel Profiling — MI350X rocprofv3 ATT Sweep & Dashboard"
author: "jhinpan (FlyDSL profiling sweep)"
url: https://jhinpan.github.io/flydsl-kernel-profiling/
source_category: benchmark-blog
architectures: [gfx950, cdna4]
tags: [flydsl, profiling, mfma, moe, gemm, attention]
retrieved_at: 2026-09-28
---

# FlyDSL Kernel Profiling — MI350X rocprofv3 ATT Sweep & Dashboard

First-party profiling study of every major FlyDSL gfx950 kernel, captured on
real MI350X (CDNA4) silicon with rocprofv3 ATT (Advanced Thread Trace) plus
hardware counters, against matched-shape AITER / CK / hipBLASLt baselines.
Each kernel ships a reproducible bundle (report + ATT trace + counters +
source) and the results are browsable as a dashboard. Method: ROCm 7.2.0,
FlyDSL 0.1.9.dev @ 18c5a7ed, `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` for source
attribution; 17 kernels profiled, 15 with baselines; ratios below are FlyDSL
throughput ÷ baseline throughput (>1 = FlyDSL faster).

## Verdicts (MI350X, gfx950)

| Bucket | Kernel | FlyDSL vs baseline | Baseline |
|---|---|---|---|
| WIN | softmax | **2.05x** | Triton |
| WIN | hgemm_splitk | **1.66x** | CK / hipBLASLt |
| WIN | moe_gemm | **1.11x** (stage2-atomic 1.30x) | AITER |
| PARITY | layernorm, quant, moe_reduce | ~1.0x | AITER |
| HEADROOM | moe_blockscale | 0.82x | tuned-CK |
| HEADROOM | rmsnorm | 0.89x | AITER |
| HEADROOM | mla (decode) | 0.90x | AITER |
| HEADROOM | flash_attn | 0.92x | CK-tile |
| HEADROOM | paged-attention | 0.48x | AITER |
| HEADROOM | topk_gating | **0.22x** | AITER |
| HEADROOM | rope | **0.17x** | AITER |

GEMM re-measured at compute-bound shapes: preshuffle 0.77x, blockscale 0.66x
vs tuned-CK; an internal v2 path is 1.20x over v1.

## Findings this wiki cites it for

- **Register-pressure-capped occupancy is the dominant headroom** on the
  attention/GEMM losers (mla, paged-attention, flash_attn, moe_blockscale):
  only 1-2 waves/SIMD resident with 175-251 VGPRs live. Cutting the live VGPR
  set to admit a second wave is the identified lever.
- **rope / topk_gating** serialize on cross-lane reductions (`shuffle_xor` /
  `ds_bpermute` against `LGKMCNT`); the suggested fix is a DPP /
  `v_permlane16`-style wave reduction.
- **The softmax fast path (`BufferCopy128b`) was dead-coded behind `False
  and`.** Re-enabling it measured on-par to +7% on large bf16 and neutral to
  slightly negative on f32 — both paths already saturate HBM (~5 TB/s) and
  register-buffer the whole row, so vectorization only trims instruction
  count. The 2.05x headline is FlyDSL-vs-Triton, not fast-vs-scalar: a
  roofline check would have predicted the small A/B delta.
- **`fp8_gemm_4wave` (rowscale) fails to compile** — `missing
  _reusable_slot_spec` on the fast-dispatch path; a config-independent
  regression, recorded as such.
