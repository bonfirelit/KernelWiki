---
id: doc-rocm-flash-attention
title: "ROCm flash-attention — CK and Triton backends"
url: https://github.com/ROCm/flash-attention
source_category: official-doc
architectures: [gfx90a, gfx942, gfx950, gfx1100, gfx1201, cdna2, cdna3, cdna4, rdna3, rdna4]
tags: [flash-attention, attention, composable-kernel, triton-rocm]
retrieved_at: 2026-09-15
---

# ROCm flash-attention

AMD's fork of flash-attention. Two backends, both implementing FlashAttention-2.

## Stable guidance used by this wiki

- **Backends**:
  - **composable_kernel (ck)** — the default backend.
  - **Triton** — provided by the `aiter` package, vendored as a submodule at
    `third_party/aiter`. Enabled with `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"`.
- **Backend selection / tuning env vars**:
  - `FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"` — use the Triton backend.
  - `FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"` — one-time autotune warmup.
  - `FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON` — a single fwd config that
    overrides defaults for `attn_fwd`.
- **Architecture support** (as stated in the README):
  - CK backend: "MI200x, MI250x, MI300x, MI355x, and RDNA 3/4 GPUs".
  - Triton backend: "AMD's CDNA (MI200, MI300) and RDNA GPUs".
- **CK backend**: dtypes fp16 and bf16; forward and backward head dimensions up
  to 256.
- **Triton backend**: dtypes fp16, bf16, and fp32; supports "forward and
  backward passes with causal masking, variable sequence lengths, arbitrary
  Q/KV sequence lengths and head sizes, MQA/GQA, dropout, rotary embeddings,
  ALiBi, paged attention, and FP8 (via the Flash Attention v3 interface)".
  Sliding-window attention is a work in progress.
- Requires ROCm 6.0 and above.
