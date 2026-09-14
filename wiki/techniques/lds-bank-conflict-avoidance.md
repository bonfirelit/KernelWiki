---
id: technique-lds-bank-conflict-avoidance
title: "LDS Bank Conflict Avoidance"
type: technique
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
tags: [lds-bank-conflict-avoidance, lds, swizzling, shared-memory-optimization]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-lds]
related: [hw-lds, technique-swizzling, pattern-lds-bank-conflicts, pattern-memory-bound, kernel-cdna4-fp8-gemm]
sources: [blog-rocm-fp8-gemm-cdna4, doc-rocm-workload-optimization, blog-rocm-memory-scheduling]
symptoms: [lds-bank-conflicts, memory-bound]
---

# LDS Bank Conflict Avoidance

The AMD counterpart of `technique-swizzling`. A conflicting LDS access does not
fail; it serializes, and the cost shows up as `s_waitcnt lgkmcnt` time that
looks like memory-bound behaviour.

## The wide-access subtlety

`ds_read_b128` is not one access. It executes in **four phases, and each phase
must independently be conflict-free** (`blog-rocm-fp8-gemm-cdna4`). A layout
that is conflict-free for `ds_read_b32` can conflict badly once you widen to
`b128` — which the ROCm ISA checklist tells you to do for bandwidth. The two
goals pull against each other, and that is exactly why a swizzle is needed
rather than just a wider type.

## Two remedies

**Padding** — add one element per row so consecutive rows start in different
banks. Cheap to write, costs LDS capacity, and the padding must survive being
widened to a vector type.

**XOR swizzle** — permute the column index by a function of the row. This costs
no LDS and is what the measured CDNA4 kernel uses: "a row-based XOR remap on
16-byte columns", `swizzled_col = c ^ mask(r)` with `mask(r) = perm << 4`. XOR
is self-inverse, so the same expression serves reads and writes.

```cpp
// XOR swizzle over 16-byte (float4) columns. Same mapping on store and load,
// because XOR is its own inverse --- no second index expression to keep in sync.
#define LDS_COLS 16   // float4 columns per row

__device__ __forceinline__ int swizzle_col(int row, int col) {
  return col ^ (row % LDS_COLS);
}

__global__ void gemm_tile_stage(const float4* __restrict__ g, float4* __restrict__ out) {
  __shared__ float4 tile[16][LDS_COLS];       // no padding: swizzle instead
  const int r = threadIdx.y, c = threadIdx.x;

  tile[r][swizzle_col(r, c)] = g[r * LDS_COLS + c];   // ds_write_b128
  __syncthreads();

  // Column-major read of the same tile: without the swizzle every lane in the
  // read would target one bank column and serialize.
  out[r * LDS_COLS + c] = tile[c][swizzle_col(c, r)]; // ds_read_b128
}
```

## Verification

There is no substitute for reading the ISA. Dump it with
`AMDGCN_ENABLE_DUMP=1` (or `hipcc --save-temps`) and check that LDS traffic is
`ds_read_b128`/`ds_write_b128` rather than a stream of `_b32`, then profile with
`rocprofv3` / ROCm Compute Profiler and look at the LDS bank-conflict counter
directly — a swizzle that is correct on paper and wrong by one shift produces
the same output and none of the speedup.

## Boundary

`blog-rocm-fp8-gemm-cdna4` measures the swizzle rung at 497.43 TFLOP/s against
506.70 for the preceding direct-global-to-LDS rung on M=N=K=4096 — i.e. the
swizzle alone was *not* a win at that point in the ladder. It became one only
after double buffering (1166.41) increased the LDS pressure the conflicts were
throttling. Sequence matters; do not adopt a swizzle on the strength of a
microbenchmark taken out of the pipeline it belongs to.
