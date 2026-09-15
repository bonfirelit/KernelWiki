---
id: doc-pytorch-sdpa-backends
title: "PyTorch scaled_dot_product_attention backend selection (sdpa_kernel)"
url: https://docs.pytorch.org/docs/2.14/generated/torch.nn.attention.sdpa_kernel.html
source_category: official-doc
architectures: [sm90, sm100, gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [attention, flash-attention]
retrieved_at: 2026-09-15
---

# PyTorch SDPA backend selection

How `torch.nn.functional.scaled_dot_product_attention` chooses an implementation,
and how to pin one. The dispatch mechanism is vendor-neutral; which concrete
backends exist behind each enum value depends on the platform (on ROCm the flash
path is AOTriton — see `doc-aotriton`).

## Stable guidance used by this wiki

- **`SDPBackend` enum members** (verbatim): `SDPBackend.FLASH_ATTENTION`,
  `SDPBackend.EFFICIENT_ATTENTION`, `SDPBackend.MATH`,
  `SDPBackend.CUDNN_ATTENTION`.
- **Forcing a backend**: `torch.nn.attention.sdpa_kernel(backends, set_priority=False)`
  is a "Context manager to select which backend to use for scaled dot product
  attention." It restricts execution to the given backend(s); on leaving the
  block "the previous state of the flags will be restored, enabling all
  backends." `set_priority` decides whether the list is read as a priority
  order.

## Boundary

The page does not state which backends are available on ROCm/HIP, and does not
itself describe `MATH` as an always-available fallback. This wiki pairs it with
`doc-aotriton` and `doc-rocm-flash-attention` for the ROCm specifics.
