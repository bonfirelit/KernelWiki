---
id: doc-cdna4-isa
title: "AMD Instinct (CDNA4) Instruction Set Architecture"
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
source_category: official-doc
architectures: [gfx950, cdna4]
tags: [mfma, lds, wave64, mxfp8, mxfp4, ocp-fp8]
retrieved_at: 2026-09-28
---

# AMD CDNA4 ISA reference (gfx950)

The official ISA manual for gfx950. The delta against `doc-cdna3-isa` that
matters for kernels:

- **Block-scaled MFMA**: `v_mfma_scale_f32_16x16x128_f8f6f4` and
  `v_mfma_scale_f32_32x32x64_f8f6f4` with E8M0 per-32-element block scales;
  `CBSZ`/`BLGP` repurposed as per-matrix format selectors
  (E4M3/E5M2/E2M3/E3M2/E2M1) and `ABID[0]` gating the scale operand.
- **OCP FP8/FP6/FP4** replaces gfx942's FNUZ encodings — not bit-compatible.
- `ds_read_tr16_b64` LDS transpose read; LDS grows to 160 KiB per CU.
- Direct-to-LDS widened to 16 B per lane (`{1,2,4,12,16}` legal sizes).
- XF32/TF32 MFMA forms removed; FP64 matrix throughput halved per CU
  relative to CDNA3 (datasheet-level).
- `v_permlane16_swap_b32` / `v_permlane32_swap_b32` cross-lane VALU ops;
  `v_smfmac_*` structured-sparsity (4:2) MAC forms.

Where the manual and silicon disagree, `blog-rocm-kernel-wiki`'s
MI350X/MI355X verification harness results take precedence in this wiki.
