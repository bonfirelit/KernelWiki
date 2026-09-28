---
id: migration-gfx942-to-gfx950
title: "Migrating a CDNA3 kernel to CDNA4 (gfx942 to gfx950)"
type: migration
from_arch: gfx942
to_arch: gfx950
tags: [ocp-fp8, mxfp4, mxfp8, fine-grained-quantization, mfma, lds, global-load-lds, cross-lane, ds-transpose]
related: [hw-mfma-cdna, hw-lds, hw-amd-memory-ops, hw-amd-narrow-precision, hw-chiplet-xcd, kernel-cdna4-fp8-gemm, migration-cdna-to-rdna4]
sources: [blog-rocm-kernel-wiki, doc-cdna3-isa, doc-cdna4-isa, doc-rocm-workload-optimization, blog-rocm-occupancy-mi355x]
amd_relevance: "MI300X (gfx942) to MI350X/MI355X (gfx950) is the generational step most production CDNA kernels are actually making, and its headline break is silent: a kernel can compile, launch, and return wrong FP8 numbers. This page is the what-breaks checklist, with items marked where blog-rocm-kernel-wiki reproduced them on MI350X silicon."
confidence: source-reported
reproducibility: pseudocode
---

# Migrating a CDNA3 kernel to CDNA4 (gfx942 to gfx950)

Rebuild with `--offload-arch=gfx950` and most HIP code runs. The danger is the
minority that compiles, launches, and computes the *wrong thing* — led by FP8.
Items marked silicon-verified were reproduced on a live MI350X by
`blog-rocm-kernel-wiki`.

## What breaks

**FP8 blobs are not portable — re-quantize, never reinterpret.** gfx942 FP8 is
FNUZ, gfx950 is OCP (`doc-cdna3-isa`, `doc-cdna4-isa`). The encodings are not
bit-compatible: FNUZ E4M3 tops out at a max finite of 240, OCP at 448, and the
exponent bias differs, so the scale that kept a tensor in range on MI300X is
wrong on MI350X. A checkpoint or KV cache quantized under FNUZ must be
re-quantized — recompiling only fixes values computed *after* the move. Select
the OCP type in hipBLASLt, not the reused FNUZ enum.

**TF32/XF32 MFMA is removed, not emulated.** gfx942's
`v_mfma_f32_16x16x8_xf32` has no gfx950 encoding: the intrinsic fails at
instruction selection ("Cannot select"), at compile time, with no BF16 fallback
silently substituted (silicon-verified, `blog-rocm-kernel-wiki`). Port to an
explicit BF16 path so the precision/throughput trade is intentional.

**FP64 matrix throughput is halved per CU** (datasheet-level) — silicon was
reallocated to the MX formats. Re-roofline FP64-matrix kernels before assuming
the newer part wins; this is an HPC item, not an ML one.

## What changes under your tuning

**LDS: 64 KiB → 160 KiB per CU, 32 → 64 banks.** Spend the headroom on deeper
prefetch, not on moving accumulators off-chip — `blog-rocm-occupancy-mi355x`
calls it "a latency-hiding budget, not an accumulator substitute". The bank
doubling invalidates gfx942 padding habits: a stride that was conflict-free on
32 banks can conflict on 64, and the `ds_read_b128` phase structure changes
from 8 phases × 8 lanes to 4 phases × 16 lanes. Re-derive swizzle masks and
padding against the new geometry, not the old constants (`hw-lds`).

**Direct-to-LDS widens.** Legal per-lane sizes go from {1,2,4} B on gfx942 to
{1,2,4,12,16} B on gfx950 — and there is no 8-byte form on *either* generation
(silicon-verified, `blog-rocm-kernel-wiki`). A staging loop built on
single-dword copies moves 4x as much per instruction as `dwordx4`. There is
still no RDNA counterpart at any width (`hw-amd-memory-ops`).

**Cross-lane gains `v_permlane16_swap_b32` / `v_permlane32_swap_b32`**
(`__builtin_amdgcn_permlane16_swap`): a 16- or 32-lane-half butterfly step with
no LDS crossbar traffic. Note the inversion — the RDNA selector form
`v_permlanex16` does *not* assemble for gfx950 (silicon-verified,
`blog-rocm-kernel-wiki`). CDNA4 exposes only the `_swap` form, so keep the
gfx942 `ds_bpermute` path and the gfx950 `_swap` path behind arch guards.

## What is new (opt-in)

**Block-scaled MFMA.** `v_mfma_scale_f32_16x16x128_f8f6f4` and
`..._32x32x64_f8f6f4` take FP8/FP6/FP4 operands with one E8M0 scale per
32-element block along K (`doc-cdna4-isa`). The legacy CBSZ/BLGP modifier
fields are repurposed as per-matrix format selectors, and ABID[0] gates which
scale slot is live — read the encoding before reusing an old macro. Toolchain
note: ROCm 7.2.4 clang exposes only the *scaled* builtins
(`blog-rocm-kernel-wiki`), so plan on carrying scales even for plain FP8.
Existing FP16/BF16/FP8 kernels keep working; the MX path is upside you opt
into (`hw-amd-narrow-precision`).

**`ds_read_tr16_b64`** — an LDS read with a built-in 16-element transpose for
MFMA operands — exists on gfx950 only: absent on gfx942, absent on RDNA4.

## Partition modes

Compute SPX/DPX/QPX/CPX; memory NPS1/NPS2. **NPS4 is not advertised on
MI350X** — it is an MI300-class (gfx942) option (silicon-verified,
`blog-rocm-kernel-wiki`). Docs pairing CPX with NPS4 describe gfx942; the
gfx950 fine-grained pairing is CPX/NPS2. See `hw-chiplet-xcd`.

## Transfers to RDNA4?

Partially — and the direction matters. gfx1201 is OCP FP8 like gfx950, so a
blob re-quantized for CDNA4 is numerically at home on RDNA4 while the gfx942
FNUZ blob never is; OCP (and the WMMA matrix core) is the shared heritage
(`hw-wmma-rdna4`). The gfx950-specific wins do not transfer: no direct-to-LDS
at any width, no `ds_read_tr16_b64`, no block-scaled matrix path — dequantize
to FP16 and use WMMA instead (`migration-cdna-to-rdna4`). The cross-lane story
inverts too: `v_permlanex16`, which refuses to assemble on gfx950, *is* the
RDNA form. And gfx1201's 64 KiB LDS matches gfx942, not gfx950 — budgets grown
for CDNA4 have to shrink back.
