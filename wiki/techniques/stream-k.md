---
id: technique-stream-k
title: "Stream-K — Flat MAC-Iteration Decomposition over a Persistent Grid"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [stream-k, persistent-kernel, tile-scheduling, split-k, xcd]
confidence: source-reported
reproducibility: snippet
prerequisites: [technique-persistent-kernels]
related: [technique-split-k, technique-persistent-kernels, pattern-tail-effect, hw-chiplet-xcd]
sources: [blog-rocm-kernel-wiki]
symptoms: [tail-effect, low-compute-utilization]
---

# Stream-K

A data-parallel GEMM grid almost never divides the CU count. With ~304 CUs on
gfx942 (256 on gfx950), a 320-tile problem runs one full wave and then a
16-tile wave while the rest of the machine drains — the `pattern-tail-effect`,
worth 20-50% of wall clock on practically sized GEMMs (`blog-rocm-kernel-wiki`).
`technique-split-k` treats the opposite symptom (too few tiles, long K) but
only re-quantizes at a fixed granularity. Stream-K removes the quantization:
decompose the **total MAC-iteration space**, not the tile grid.

## The decomposition

Flatten the GEMM to `total_iters = num_tiles · ceil(K/BK)` loop iterations.
Launch a persistent grid of **exactly `numCUs` workgroups** (one resident per
CU — this is why `technique-persistent-kernels` is the prerequisite) and hand
each a contiguous span `[cu·q, min(cu·q + q, total))` with
`q = ceil(total / numCUs)`. A span generally starts mid-tile and ends mid-tile,
so a workgroup may contribute a *partial* to the tiles at its span's ends.

**Fix-up rule**: the workgroup that owns a tile's *first* iteration is
responsible for that tile — it collects the other contributors' partials
(semaphore-guarded workspace) and stores the final tile. Everyone else
publishes a partial and moves on.

```cpp
#include <hip/hip_runtime.h>

// Scheduler core of a stream-K walk. The MFMA mainloop inside is unchanged
// from a data-parallel GEMM; only the iteration bookkeeping is new.
__global__ void streamk_walk(int iters_per_tile, int total_iters,
                             int iters_per_cu, int* __restrict__ touched) {
  int it     = blockIdx.x * iters_per_cu;
  int it_end = it + iters_per_cu;
  if (it_end > total_iters) it_end = total_iters;

  int n = 0;
  while (it < it_end) {
    const int tile    = it / iters_per_tile;   // tile this span enters
    const int k_first = it % iters_per_tile;   // first K-iter we own
    int tile_end      = (tile + 1) * iters_per_tile;
    if (tile_end > it_end) tile_end = it_end;
    // k_first == 0 -> we own this tile's FIRST iteration: run the
    // semaphore-guarded fix-up, folding in the other contributors.
    // k_first  > 0 -> publish our partial to the workspace and move on.
    (void)k_first;
    it = tile_end;                             // advance past the run we own
    ++n;
  }
  if (threadIdx.x == 0) touched[blockIdx.x] = n;
}
```

## Determinism posture — state it precisely

The iteration→workgroup assignment is a pure function of the launch
configuration, so the *decomposition* is deterministic. Whether the *result*
is bit-reproducible depends entirely on the fix-up: atomic-add fix-up reorders
FP summation and is not bit-reproducible run to run; a per-tile semaphore with
a fixed-order reduction is bit-reproducible **for a fixed launch config** —
change `numCUs` or the tile size and the summation order changes with it
(`blog-rocm-kernel-wiki`). If you need bit-identical output across machines,
stream-K is the wrong tool.

## What production ships

Not pure stream-K but the **hybrid**: the bulk of the grid runs data-parallel,
and only the ragged tail — the tiles that would form the partial wave — is
decomposed stream-K style (`blog-rocm-kernel-wiki`). That kills the tail
without paying fix-up costs on tiles that never needed them, and without
split-K's determinism loss on the DP portion. On chiplet parts, rasterize the
flat tile index so consecutive workgroups land on tiles sharing A-rows or
B-columns — L2 is per-XCD, so the mapping decides whether shared operands are
served once or replicated across chiplets (`hw-chiplet-xcd`,
`pattern-xcd-locality`).

## Pitfalls

- The persistent grid pins one workgroup per CU; VGPR/LDS use must still allow
  a resident workgroup per CU or you lose the parallelism you launched for
  (check with the math in `technique-occupancy-tuning-amd`).
- Workspace fix-up needs scratch for the worst case, `numCUs · BM · BN`
  accumulators; size it or fall back to atomics.
- Prefer plain data-parallel when the tile count is already a large multiple
  of `numCUs` — the fix-up is pure overhead there.

## Transfers to RDNA4?

**Yes.** The flat decomposition and the first-iteration-owner fix-up are
scheduler logic, oblivious to whether the inner loop is MFMA or WMMA. Two
RDNA4-specific notes: there is no XCD dimension, so the rasterization argument
collapses to ordinary L2 locality; and the persistent grid must be sized from
the 64 CUs, not the 32 WGPs that `multiProcessorCount` reports (`hw-gfx1201`).
