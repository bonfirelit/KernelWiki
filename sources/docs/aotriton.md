---
id: doc-aotriton
title: "AOTriton — Ahead-of-Time Triton math library (ROCm)"
url: https://github.com/ROCm/aotriton
source_category: official-doc
architectures: [gfx942, gfx950, gfx1100, gfx1201, cdna3, cdna4, rdna3, rdna4]
tags: [flash-attention, attention, triton-rocm]
retrieved_at: 2026-09-15
---

# AOTriton

The ahead-of-time-compiled Triton attention library that PyTorch's SDPA flash
path uses on ROCm. Repo `ROCm/aotriton`, MIT license.

## Stable guidance used by this wiki

- **What it is**: an "Ahead of Time (AOT) Triton Math Library". Its first
  supported kernel is FlashAttention, "based on the algorithm from Tri Dao".
  Precompiled binaries ship with PyTorch and are consumed through the SDPA
  kernels; the integration file named by the README is
  `aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip`, with
  download logic in `cmake/External/aotriton.cmake`.
- **Why ahead-of-time**: the kernels are compiled to an archive that a project
  links against, with no Triton needed at the consumer's build/run time. On
  Windows, where Triton is unavailable, the documented path is to set
  `AOTRITON_NOIMAGE_MODE` and reuse the `aotriton.images` folder from a Linux
  build — i.e. kernel images are built offline and shipped as data.
- **Tuning database** (release 0.12b, 2026-05-18): updated for
  `gfx942, gfx950, gfx1100, and gfx1201` (#172), and sharded into per-arch
  files under `v3python/database/<vendor>/<arch>/` (#133). An unsupported
  `AOTRITON_TARGET_ARCH` now fails loudly. The 0.13.50tp preview (2026-07-23)
  added a mini tuning database for `gfx1250`.
- **Archive artifacts**: images ship per-arch, e.g.
  `aotriton-0.13b-images-amd-gfx942.tar.gz`; `.aks2` kernel archives are stored
  in "uncompressed zip containers" as of 0.13b.
- **Per-architecture known problems (0.12b)**:
  - `gfx1201`: "a small number of unit tests fail due to a hipblasLt GPU
    segfault".
  - `gfx1100`: "a small number of unit tests fail due to compiler accuracy
    issues".
  - `gfx950`: "hdim=48/80 backward kernels disabled pending a compiler fix";
    hdim=16 forward rounds up to hdim=32.
- **PyTorch compatibility**: PyTorch 2.9 -> AOTriton 0.11b/0.10b; 2.8 ->
  0.10b/0.9b; older maps down to 2.3 -> 0.4b.

## Boundary

An open FlyDSL / "flyc" attention backend is being integrated per PRs #227 and
#230, but neither the README nor the release notes captured here establish its
behaviour, so this wiki makes no claim about it.
