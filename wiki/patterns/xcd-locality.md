---
id: pattern-xcd-locality
title: "Cross-XCD traffic: tile neighborhoods scattered across chiplet L2s"
type: pattern
tags: [xcd, tile-scheduling, persistent-kernel]
symptoms: [memory-bound, low-compute-utilization, xcd-remote-traffic, tail-effect]
candidate_techniques: [technique-persistent-kernels, hw-chiplet-xcd, technique-stream-k]
related: [hw-chiplet-xcd, pattern-memory-bound, pattern-tail-effect, technique-persistent-kernels, kernel-cdna4-fp8-gemm]
sources: [blog-rocm-kernel-wiki, blog-amdgpu-kernel-opt-guide, doc-rocm-workload-optimization, blog-rocm-occupancy-mi355x]
architectures: [gfx942, gfx950, cdna3, cdna4]
confidence: source-reported
---

# Cross-XCD traffic: tile neighborhoods scattered across chiplet L2s

## Symptom

A GEMM or attention kernel sits well below the HBM roofline even though loads
are already `global_load_dwordx4` and LDS is swizzled — the usual
`pattern-memory-bound` rungs are climbed. The L2 hit rate is low for a working
set that fits comfortably in aggregate L2 (8 × 4 MiB), and — the telltale —
the hit rate is **asymmetric across XCDs**: some slices run hot while others
miss constantly, with fabric/EA read traffic to match. The tiles are sized
right; they are landing in the wrong cache.

## Likely cause

Workgroup dispatch to XCDs is round-robin (`blockIdx % 8`), so a linear tile
walk deals every operand-sharing neighborhood across all eight private 4 MiB
L2s — each XCD refetches the same A-panel or B-panel into its own slice
(dispatch model source-reported from `blog-amdgpu-kernel-opt-guide`; topology
in `hw-chiplet-xcd`). MALL absorbs some of the refetch, but a line two XCDs
both want is fetched per XCD.

## Candidate techniques

| Technique | When |
|---|---|
| `technique-persistent-kernels` | First stop. One workgroup per CU; remap the linear tile id so each XCD owns a contiguous *band* of output tiles whose shared operands stay resident in its 4 MiB L2. Undo the round-robin: `tile = xcd * tiles_per_xcd + slot`. |
| `hw-chiplet-xcd` | Background: topology, the dispatch model, and the CPX/NPS partition fallback for kernels that cannot change. |
| `technique-stream-k` | When the tile count does not divide into 304/256 CUs: run Stream-K *within* an XCD's band so K-partials reduce inside one L2 instead of spraying across the fabric. |

Order each band so adjacent tiles share the reused operand — the same idea as
a threadblock rasterization swizzle, but the locality unit is the XCD's L2,
not an SM's L1.

## Diagnosis

- `rocprofv3 --pmc TCC_HIT TCC_MISS TCC_EA_RDREQ` — low hit rate plus high EA
  (off-XCD fabric) read requests relative to compute is the signature
  (`doc-rocm-workload-optimization`).
- **Single-XCD pinning experiment**: under CPX each logical device *is* one
  XCD. If the single-XCD hit rate jumps far above the full-chip rate at an
  unchanged 4 MiB, locality — not capacity — is the binding problem. Cheapest
  definitive test, and it needs no kernel changes.

## Caveats

- **4 MiB per XCD is the budget.** A band whose reuse set exceeds one slice
  spills to MALL anyway; tune band width against 4 MiB, not the 256 MiB MALL.
- **Persistent kernels hold registers and LDS for their lifetime** — watch
  occupancy so the locality win is not eaten by a VGPR regression.
- **Cross-XCD atomics serialize** through the coherence point; keep partial
  reductions XCD-local.
- Round-robin assumes 8 XCDs in SPX; query the partition mode at runtime
  rather than hardcoding (`hw-chiplet-xcd`).

## Transfers to RDNA4?

No. gfx1201 has a single die and a single shared L2, so there is no chiplet
boundary for tiles to cross (`hw-gfx1201`). File the banded-rasterization idea
under ordinary L2-friendly swizzling — still useful there, but not a distinct
pattern.
