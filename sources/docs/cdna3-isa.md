---
id: doc-cdna3-isa
title: "AMD Instinct MI300 (CDNA3) Instruction Set Architecture"
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
source_category: official-doc
architectures: [gfx942, cdna3]
tags: [mfma, lds, wave64]
retrieved_at: 2026-09-28
---

# AMD CDNA3 ISA reference (gfx942)

The official ISA manual for gfx942: MFMA encodings, `ds_*` LDS ops, buffer /
global / flat memory instructions, `s_waitcnt` fields, and the shader
programming model. Used here as the primary reference for CDNA3 instruction
semantics (phase behavior, operand layouts, waitcnt bitfields).

Key facts taken from it:

- MFMA shape menu for gfx942, including INT8 (`v_mfma_i32_16x16x32_i8`,
  `v_mfma_i32_32x32x16_i8`) and the XF32/TF32 forms (`v_mfma_f32_16x16x8_xf32`,
  `v_mfma_f32_32x32x4_xf32`) that gfx950 later drops.
- `s_waitcnt` immediate layout: vmcnt[3:0], expcnt[6:4], lgkmcnt[12:8] —
  the bitfield map every inline-asm wait count is checked against.
- LDS banked organization (32 banks) and `ds_read2_*` / `ds_read_b128`
  phased execution model.
