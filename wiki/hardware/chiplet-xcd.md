---
id: hw-chiplet-xcd
title: "XCD chiplets: per-XCD L2, MALL, and partition modes"
type: hardware
architectures: [gfx942, gfx950, cdna3, cdna4]
tags: [xcd, tile-scheduling, persistent-kernel]
confidence: source-reported
related: [pattern-xcd-locality, hw-lds, hw-mfma-cdna, technique-persistent-kernels]
sources: [blog-rocm-kernel-wiki, blog-amdgpu-kernel-opt-guide, doc-cdna3-isa, doc-cdna4-isa, blog-rocm-occupancy-mi355x]
aliases: [XCD, chiplet, "Infinity Cache", MALL]
---

# XCD chiplets: per-XCD L2, MALL, and partition modes

MI300X/MI350X-class parts are not monolithic GPUs. The compute silicon is 8
**XCDs** (Accelerator Complex Dies) stacked over I/O dies, and each XCD owns
its CUs plus a **private 4 MiB L2**. There is no device-wide L2: the GPU is a
NUMA machine, and a kernel that ignores the boundary pays inter-die fabric
prices for what it thought was cache reuse.

| Part | gfx | XCDs | Active CUs | CUs per XCD | Per-XCD L2 |
|---|---|---|---|---|---|
| MI300X | gfx942 | 8 | 304 | 38–40 | 4 MiB |
| MI350X/MI355X | gfx950 | 8 | 256 | 32 | 4 MiB |

Occupancy math must use the *active* CU count — a few CUs per XCD are fused
off for yield. `blog-rocm-kernel-wiki` confirmed 256 CUs / gfx950 on a live
MI355X.

Below the XCDs sits the memory side: 8 HBM stacks, each fronted by a **32 MiB
MALL (Infinity Cache) slice — 256 MiB total**. MALL absorbs memory-side
traffic but is not a substitute for keeping reuse inside one XCD's L2: a line
resident in XCD0's 4 MiB is a fabric round-trip away from a consumer on XCD1,
and cross-XCD traffic is exactly that — over the inter-die fabric, not a
shared cache lookup.

## Dispatch order is the trap

Workgroups are dealt to XCDs **round-robin** — block `b` lands on XCD
`b % 8` — so a GEMM that walks output tiles in linear `blockIdx` order
scatters every operand-sharing tile neighborhood across all eight private L2s.
This dispatch-order claim is **source-reported** from
`blog-amdgpu-kernel-opt-guide`; even `blog-rocm-kernel-wiki` did not
runtime-verify it. Treat it as the working model regardless — the remedy
(`pattern-xcd-locality`) is cheap and pays off exactly when the model holds.

The diagnostic is **per-XCD L2 hit-rate asymmetry**: TCC_HIT/TCC_MISS counters
broken out per XCD, with some slices hot and others missing constantly, plus
elevated fabric read traffic. `pattern-xcd-locality` has the full
differential.

## Partition modes

Firmware carves the machine along two independent axes (`amd-smi partition`):

- **Compute**: SPX (one device, 8 XCDs), DPX (2×4), QPX (4×2), **CPX** (8×1 —
  each XCD is its own logical GPU).
- **Memory**: NPS1 (all HBM one interleaved pool) or NPS2 (two NUMA halves).
  **NPS4 is not advertised on MI350X** — it is an MI300-class (gfx942) layout;
  silicon-verified by `blog-rocm-kernel-wiki`. Docs pairing CPX with NPS4
  describe gfx942.

CPX makes the NUMA effect disappear *inside* a device — the per-XCD L2 becomes
"the" L2 — at the cost of an eighth of the CUs per device. For kernels you
cannot modify, CPX plus an NPS split is the coarse locality fallback. For a
single large GEMM, stay in SPX/NPS1 and fix the tile schedule in software
(`technique-persistent-kernels`).

## Practical notes

- **Query, don't hardcode.** CU counts differ across gfx942/gfx950 and shrink
  under CPX; read them from `hipGetDeviceProperties` at runtime.
- **Atomics hammered from all XCDs serialize** through the coherence point —
  prefer per-XCD partials with a second-pass reduction.
- **LDS plays no role in the XCD story**: it is per-CU, explicitly managed
  workgroup storage (`hw-lds`), not a transparent cache level. Only L2 is
  per-XCD; MALL is per-stack.

## Transfers to RDNA4?

No. gfx1201 is a single-die part — one 8 MiB L2 behind 4 shader engines, no
chiplet boundary, no XCD concept (`hw-gfx1201`). What generalizes is the
dispatch-order awareness: keeping operand-sharing tiles co-resident in *some*
cache level is worth doing anywhere, but on RDNA4 that level is the single
shared L2 and an ordinary rasterization swizzle suffices.
