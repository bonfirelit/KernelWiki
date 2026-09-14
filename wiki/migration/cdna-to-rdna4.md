---
id: migration-cdna-to-rdna4
title: "Migrating a CDNA kernel to RDNA4 (gfx1201)"
type: migration
from_arch: gfx942
to_arch: gfx1201
tags: [mfma, wmma, agpr, global-load-lds, ds-transpose, lds, wave32, wave64]
related: [hw-gfx1201, hw-wmma-rdna4, hw-mfma-cdna, hw-lds, hw-amd-memory-ops, technique-in-register-transpose, technique-occupancy-tuning-amd, kernel-rdna4-wmma-gemm, kernel-cdna4-fp8-gemm]
sources: [doc-amd-rdna4-matrix-cores, blog-salykova-matrix-cores-cdna, blog-rocm-fp8-gemm-cdna4, blog-rocm-occupancy-mi355x, blog-rdna4-wmma-lane-mapping, blog-rocm-mxfp4-rotation, doc-rocm-workload-optimization]
amd_relevance: "Nearly all published AMD kernel-optimization writing targets CDNA (MI300X/MI355X), while RDNA4 is the target most developers can actually buy. This page is the accounting of which CDNA techniques survive the move and which have no gfx12 counterpart at all."
confidence: source-reported
reproducibility: pseudocode
---

# Migrating a CDNA kernel to RDNA4 (gfx1201)

The two families share a toolchain, an assembly syntax, and a memory model.
They do not share a matrix core, a register-file structure, or an asynchronous
copy path. A CDNA GEMM does not "port with tuning" — parts of it have to be
rebuilt.

## Required changes

| CDNA3/4 mechanism | gfx1201 | What to do instead |
|---|---|---|
| **wave64** | **wave32** | Rewrite every lane-index derivation, ballot mask, and wave reduction. A `__shfl` mask of `0xFFFFFFFF` covers a full wave on RDNA4 and half a wave on CDNA. |
| **MFMA family** (`v_mfma_*`, many MxNxK shapes) | **WMMA**, `16x16x16` only | `__builtin_amdgcn_wmma_*_w32_gfx12`. Shape selection disappears as a tuning dimension; tiling has to be rebuilt on a fixed 16x16 tile. |
| **AGPR accumulator file** (`.agpr_count: 256`) | none | Accumulators occupy ordinary VGPRs, so they compete with operand staging for the same file. Expect to carry a smaller C tile per wave. |
| **Direct global-to-LDS** (`llvm.amdgcn.raw.buffer.load.lds`, 128 bits/lane on CDNA4) | none | HBM -> VGPR -> LDS explicitly. The in-flight VGPRs count against occupancy for the whole transfer, which pushes toward smaller K-tiles. |
| **`ds_read_tr16_b64`** LDS transpose read | none | Transpose in the LDS write pattern or in registers — `technique-in-register-transpose`. |
| **Block-scaled MFMA** (`v_mfma_scale_f32_*_f8f6f4`), native FP6/FP4 | FP8/int8/iu4 WMMA; **no native FP4 matrix input** | Dequantize to FP16 and issue FP16 WMMA — `kernel-rdna4-wmma-gemm` measures 40.8 TFLOPS doing exactly this. |
| **160 KiB LDS per CU** (CDNA4) | **64 KiB** | Re-derive every LDS budget. Deep prefetch staging sized for CDNA4 will not fit. |
| **4 SIMDs per CU**, occupancy formula `* 4` | **2 SIMDs per CU** | The `occ_vgpr * 4` term in `doc-rocm-workload-optimization`'s formula is wrong on gfx1201. |
| FP8 **FNUZ** on CDNA3 / **OCP** on CDNA4 | OCP | Numerics change silently across all three. Re-validate, do not just recompile. |

## What does transfer

- The **optimization ordering** from `kernel-cdna4-fp8-gemm`: vectorize loads
  first (11.2x there, the largest single step), then tile through LDS, then
  double-buffer, and only then hand-schedule.
- **`global_load_dwordx4` / `ds_read_b128`** as the target instruction widths, and
  the ISA-reading checklist in `lang-amdgcn-asm`.
- **Bank-conflict avoidance**, including the four-phase `ds_read_b128` subtlety
  and the XOR-swizzle remedy.
- **`sched_barrier`, `sched_group_barrier`, `s_setprio`** all compile for gfx1201
  — verified on this host. The *sequences* do not transfer, because they were
  built to interleave MFMA against direct-to-LDS traffic.
- **`s_waitcnt vmcnt/lgkmcnt` discipline** and the two-barrier double-buffer
  protocol, unchanged.

## Porting sketch

```
CDNA3/4 inner loop                        gfx1201 inner loop
------------------------------------      ------------------------------------
buffer_load_lds  -> A_lds (async)         global_load_dwordx4 -> v[..]  (VGPR)
                                          ds_write_b128        -> A_lds
s_waitcnt vmcnt(n)                        s_waitcnt vmcnt(n)
s_barrier                                 s_waitcnt lgkmcnt(0) ; s_barrier
ds_read_tr16_b64 -> operand (transposed)  ds_read_b128 -> operand
                                            (transpose folded into the write
                                             index, or shuffled in registers)
v_mfma_f32_16x16x32_bf16 -> AGPR acc      v_wmma_f32_16x16x16_f16 -> VGPR acc
s_barrier                                 s_barrier
```

The extra `ds_write_b128` and the loss of the AGPR file are the two changes that
cost the most: the first adds LDS write bandwidth and a wait the CDNA version
does not pay, the second shrinks the accumulator tile that can stay resident.

## Verify before trusting

The WMMA fragment layout is where ported code goes silently wrong: lane index
selects the **column**, not the row (`hw-wmma-rdna4`). A CDNA-shaped assumption
compiles, runs, and returns a transposed tile. Test against a non-symmetric
reference matrix.
