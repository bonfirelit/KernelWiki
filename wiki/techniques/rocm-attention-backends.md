---
id: technique-rocm-attention-backends
title: "Choosing and Tuning the SDPA / FlashAttention Backend on ROCm"
type: technique
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
tags: [attention, flash-attention, triton-rocm, composable-kernel, hip]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-gfx1201]
related: [lang-triton-rocm, lang-composable-kernel, lang-hip, hw-gfx1201, pattern-compute-bound, pattern-memory-bound]
sources: [doc-aotriton, doc-rocm-flash-attention, doc-pytorch-sdpa-backends]
symptoms: [low-compute-utilization, memory-bound]
---

# Choosing and Tuning the SDPA / FlashAttention Backend on ROCm

Attention on AMD is not one kernel — it is three separate code paths that reach
the same `scaled_dot_product_attention` call, with different maturity per
architecture. Getting the fast path is mostly a matter of picking the right one
and confirming you actually got it, not of hand-writing a kernel.

## The three paths

| Path | What it is | Reaches you through |
|---|---|---|
| **AOTriton** | Ahead-of-time-compiled Triton FlashAttention, shipped precompiled with PyTorch | `SDPBackend.FLASH_ATTENTION` on ROCm (`doc-aotriton`) |
| **Composable Kernel (CK)** | The default backend of ROCm `flash-attention`; FlashAttention-2 in C++ templates | the `flash-attn` package / `SDPBackend.EFFICIENT_ATTENTION`-style path (`doc-rocm-flash-attention`) |
| **Triton (aiter)** | The `aiter`-provided Triton backend of ROCm `flash-attention` | `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` |

For plain `torch.nn.functional.scaled_dot_product_attention`, AOTriton is the one
that matters: it is what the `FLASH_ATTENTION` backend dispatches to on ROCm,
and it is already compiled into your PyTorch wheel.

## Pin the backend, then verify you got it

The failure mode that wastes the most time is silently running on
`SDPBackend.MATH` — the correct-but-slow reference path — and concluding "flash
attention is slow on AMD". Force the backend and make the alternative fail loudly
instead of degrading quietly:

```python
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

# Restrict to the flash path. On leaving the block PyTorch restores the previous
# flags and re-enables every backend (doc-pytorch-sdpa-backends).
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

# Diagnosis: if MATH is the only backend that runs your shape, forcing FLASH
# raises instead of silently falling back --- which is the signal you want.
def assert_flash(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
```

`SDPBackend` has exactly four members — `FLASH_ATTENTION`, `EFFICIENT_ATTENTION`,
`MATH`, `CUDNN_ATTENTION` — and `CUDNN_ATTENTION` is NVIDIA-only, so on ROCm the
real choice is flash vs. efficient vs. the math fallback
(`doc-pytorch-sdpa-backends`).

## The ROCm flash-attention package (CK vs Triton)

If you use the `flash-attn` package directly rather than PyTorch SDPA, the
backend is chosen at import/run time (`doc-rocm-flash-attention`):

```bash
# Default is Composable Kernel. Opt into the aiter Triton backend explicitly:
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"   # one-time warmup search
# Pin a single forward config instead of autotuning:
export FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON=/path/to/attn_fwd.json
```

- **CK** (default): fp16/bf16, forward and backward head dims up to 256, supports
  MI200x/MI250x/MI300x/MI355x and RDNA 3/4.
- **Triton/aiter**: fp16/bf16/fp32, causal + varlen + MQA/GQA + dropout + rotary
  + ALiBi + paged attention + FP8 (the FA-v3 interface); sliding-window is a work
  in progress.

## Per-architecture reality check

The path exists on your target; whether it is complete is per-arch. From AOTriton
0.12b's known-problems list (`doc-aotriton`):

- **gfx1201 (RDNA4)**: a small number of unit tests fail "due to a hipblasLt GPU
  segfault". The FlashAttention path works, but treat backward and edge head
  dims as needing verification on this target.
- **gfx1100 (RDNA3)**: some unit tests fail on "compiler accuracy issues".
- **gfx950 (CDNA4)**: hdim=48/80 backward kernels are "disabled pending a
  compiler fix", and hdim=16 forward rounds up to hdim=32.

The tuning database is sharded per arch under
`v3python/database/<vendor>/<arch>/` and was updated for
`gfx942, gfx950, gfx1100, and gfx1201` in 0.12b. An untuned arch runs, but not
at the tuned config.

## Measured on this host

On this box (2x gfx1201, ROCm 7.2), a MiniMax-H3 t2va workload runs SDPA on the
AOTriton flash path at ~52 TFLOP/s; the `MATH` fallback for the same kernel is
~2.7 TFLOP/s and OOMs past a 16k sequence length. Two takeaways that generalize:
the ~19x gap is the entire reason to pin `FLASH_ATTENTION`, and on this host the
ROCm SDPA flash path is genuinely selected — it is not silently stuck on the math
fallback. (Host observation from the operator's environment notes, not a
corpus-sourced benchmark.)

## Transfers across AMD targets?

The *selection* mechanism (`sdpa_kernel`, the env vars) is identical everywhere.
The *backend availability* is not: CK covers CDNA and RDNA3/4; AOTriton tunes
gfx942/gfx950/gfx1100/gfx1201 but with the per-arch gaps above. Always confirm
the flash path on the exact `gfx` target rather than assuming a CDNA result
carries to RDNA4.
