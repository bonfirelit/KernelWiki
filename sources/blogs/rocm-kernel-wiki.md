---
id: blog-rocm-kernel-wiki
title: "ROCmKernelWiki — CDNA kernel optimization KB with MI350X silicon verification"
author: "jhinpan (community)"
url: https://github.com/jhinpan/ROCmKernelWiki
source_category: community-note
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [mfma, lds, xcd, global-load-lds, mxfp8, occupancy-tuning, cross-lane]
retrieved_at: 2026-09-28
---

# ROCmKernelWiki (with VERIFICATION.md / hardware-verified.yaml)

Community CDNA3/CDNA4 kernel-optimization knowledge base modeled on this
repo's schema. Its distinguishing asset is `VERIFICATION.md` plus
`data/hardware-verified.yaml`: claims checked by compiling, running, and
disassembling on a real MI350X (gfx950, ROCm 7.2), re-checked on MI355X
(ROCm 7.1.1), with each finding re-run by an adversarial second pass.

## Silicon-verified facts this wiki cites it for

- **32 waves/CU max** on gfx950 (rocminfo), not the 40 some docs imply.
- **Direct-to-LDS async copy widths**: legal per-lane sizes `{1,2,4}` B on
  gfx942 and `{1,2,4,12,16}` B on gfx950; no 8-byte form on either.
- **Cross-lane on gfx950**: `v_permlane16_swap_b32` / `v_permlane32_swap_b32`
  via `__builtin_amdgcn_permlane16_swap`; the RDNA selector form
  `v_permlanex16` does *not* assemble for CDNA4.
- **Partition modes**: compute SPX/DPX/QPX/CPX; memory NPS1/NPS2 — NPS4 is
  not advertised on MI350X (it is an MI300-class option).
- **XF32/TF32 MFMA hard-removed on gfx950**: `mfma_f32_16x16x8_xf32` compiles
  for gfx942 but fails instruction selection ("Cannot select") on gfx950.
- **LDS**: 64 banks on gfx950 (32 on gfx942) with the b32/b64 phase groups
  reproduced empirically; the b128 four-group table could *not* be reproduced
  on-device and is recorded as upstream-empirical/inconclusive. Allocation
  granularity 512 B (gfx942) / 1280 B (gfx950).
- **Buffer-descriptor OOB semantics**: raw-buffer OOB loads return 0 (stores
  drop), per-dword-component for vector widths; gfx9 descriptor word-3 flags
  must be `0x00020000` (flags=0 marks every lane OOB); gfx1201 uses
  `0x31004000`.
- **VGPR allocation granularity 8** on both gfx942 and gfx950, and the ROCm 7
  `.vgpr_count` metadata already includes the AGPR allocation.
- OCP-E4M3 max finite 448 vs FNUZ-E4M3 max 240 — the gfx942→gfx950 FP8
  re-encoding break, quantified.

## Caveats when consuming it

Its HIP/CK code snippets are weaker than its prose: a `sched_barrier` mask
polarity inversion (LLVM's AMDGPUIGroupLP treats set bits as classes *allowed*
to cross), wrong `s_waitcnt` magic constants, a single-barrier double-buffer
schedule that races, and an RDNA4 WMMA section that copies the RDNA3 operand
signature (gfx12 uses 8 elements/lane, and the `_gfx12`-suffixed builtin).
Its `ds_read_b128` four-phase table carries its own inconclusive flag. Verify
snippets against a compiler before reusing them.
