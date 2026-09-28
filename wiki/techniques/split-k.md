---
id: technique-split-k
title: "Split-K (GlobalSplitU) — Parallelizing the K Reduction"
type: technique
architectures: [gfx942, gfx950, gfx1201, cdna3, cdna4, rdna4]
tags: [split-k, tile-scheduling, mfma, occupancy-tuning, xcd]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma-cdna]
related: [technique-stream-k, pattern-tail-effect, pattern-low-sm-utilization, hw-chiplet-xcd]
sources: [blog-rocm-kernel-wiki]
symptoms: [tail-effect, low-compute-utilization]
---

# Split-K (GlobalSplitU)

A tiled GEMM launches `ceil(M/BM) · ceil(N/BN)` workgroups, each owning one
output tile and looping over all of K. When **M·N is small relative to the
machine and K is large** — decode-time projections, GEMV-like shapes, LoRA —
that product is smaller than the CU count (304 on gfx942, 256 on gfx950
(`blog-rocm-kernel-wiki`)). The kernel is then neither compute- nor
memory-bound; it is *starved*: most CUs idle while a handful stream the entire
K dimension.

Split-K (Tensile's `GlobalSplitU`, CK's `k_batch`) fixes this by partitioning
the contraction itself: slice K into `SplitK` ranges, launch `tiles · SplitK`
workgroups, and reduce the partial sums.

## Two epilogue designs

- **Atomic**: every slice accumulates straight into `C` with
  `global_atomic_add_f32`. Cheap and launch-free, but the accumulator must be
  FP32 — even for FP8/BF16 GEMMs — because that is the only atomic type worth
  trusting here (the matrix core already accumulates in FP32; keep it there).
  Addition order is whatever the hardware does, so results are **not
  bit-reproducible** run to run.
- **Workspace + reduction kernel**: each slice writes a private
  `[SplitK, M, N]` partial; a second pass reduces it. Deterministic, at the
  price of the workspace and one extra launch.

```cpp
#include <hip/hip_runtime.h>

// Atomic variant, epilogue only. gridDim.z == SplitK; each z-slice ran the
// ordinary MFMA mainloop over its K range and holds an FP32 partial tile.
// C must be pre-zeroed (beta == 0) — forgetting this corrupts silently.
template <int BM, int BN>
__global__ void splitk_epilogue(const float* __restrict__ partial, // [BM*BN]
                                float* __restrict__ C, int M, int N) {
  const int row = blockIdx.y * BM + threadIdx.y;
  const int col = blockIdx.x * BN + threadIdx.x;
  if (row < M && col < N)
    atomicAdd(&C[(size_t)row * N + col],
              partial[threadIdx.y * BN + threadIdx.x]);   // global_atomic_add_f32
}
```

## Sizing rule

Pick the **smallest** `SplitK` such that `tiles · SplitK ≥ ~numCUs`
(`blog-rocm-kernel-wiki`). Past that point every extra slice is pure reduction
cost: atomic contention on hot `C` lines (worse across XCDs, where L2 is not
shared — see `hw-chiplet-xcd`) or a fatter reduction pass. Two corollaries:
keep each slice's K long enough that the MFMA pipeline still reaches steady
state (see `technique-mfma-pipelining`), and prefer power-of-two splits so the
remainder handling stays even.

`SplitK = 1` is an ordinary GEMM. If the base tile count already covers the
CUs, split-K only adds overhead — the failure mode it treats is
`pattern-low-sm-utilization`, not a slow mainloop.

## Production answer

hipBLASLt's heuristic already selects `GlobalSplitU` per shape — expose a
workspace budget (`HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`, sized for the
largest split it may pick, or it silently falls back to `SplitK=1`) and let it
choose. Hand-roll split-K to learn, or when you can measure the library
leaving performance on the table. If a fixed split still leaves a ragged final
wave, `technique-stream-k` is the generalization.

## Transfers to RDNA4?

**Yes, mechanically.** The decomposition is matrix-unit-agnostic: same tile
grid, same K partition, same two epilogues — only the inner loop is WMMA
(wave32) instead of MFMA. The sizing rule needs the right CU count, and
gfx1201 has a trap here: `multiProcessorCount` reports 32 **WGPs**, not the 64
CUs (`hw-gfx1201`) — size the split from the CU number or you under-split by
2x.
