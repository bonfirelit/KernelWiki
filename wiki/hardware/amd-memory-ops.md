---
id: hw-amd-memory-ops
title: "AMD global memory ops, direct-to-LDS, and wait counters"
type: hardware
architectures: [gfx1201, gfx942, gfx950, rdna4, cdna3, cdna4]
tags: [global-load-lds, lds, vectorized-loads, double-buffering]
confidence: source-reported
related: [hw-lds, hw-gfx1201, hw-mfma-cdna, technique-double-buffering, technique-vectorized-loads, kernel-cdna4-fp8-gemm]
sources: [doc-rocm-workload-optimization, doc-llvm-amdgpu-usage, blog-rocm-fp8-gemm-cdna4, blog-rocm-memory-scheduling]
aliases: [global_load_lds, buffer_load_lds, "direct-to-LDS", vmcnt, lgkmcnt, s_waitcnt]
---

# AMD global memory ops, direct-to-LDS, and wait counters

The AMD counterpart to the question "how do I get a tile from HBM into fast
memory without stalling". There is no TMA and no descriptor object; the moving
parts are the load instruction width, an optional direct-to-LDS path, and two
hardware counters you drain by hand.

## Programming contract

```cpp
// Get the widest load the ISA offers. The ROCm workload guide's ISA checklist
// is literally "confirm global_load_dwordx4 is used" --- a scalar float load
// per lane leaves 4x of the request width on the floor.
__global__ void copy_wide(const float4* __restrict__ src, float4* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];   // -> global_load_dwordx4 / global_store_dwordx4
}
```

## The two counters

- **`vmcnt`** — outstanding vector-memory (global / buffer) operations.
- **`lgkmcnt`** — outstanding LDS, GDS, constant, and message traffic.

`s_waitcnt vmcnt(n)` and `s_waitcnt lgkmcnt(n)` block until at most `n` are
still in flight, so the *non-zero* forms are the whole point: a software
pipeline issues N loads and then waits on `vmcnt(N-1)` to consume the first
while the rest remain outstanding. Waiting on `(0)` everywhere is correct and
slow. Reading these operands out of the ISA dump is the documented way to see
whether the compiler pipelined your loop (`doc-rocm-workload-optimization`).

## Direct global-to-LDS (CDNA only)

CDNA can move data from global memory into LDS without landing it in VGPRs
first, reached through `llvm.amdgcn.raw.buffer.load.lds`
(`llvm_amdgcn_raw_buffer_load_lds` in HIP). CDNA4 widens the per-lane
`GLOBAL_LOAD_LDS` transfer to "Up to 128 bits/lane" against 32-bit on CDNA3.

In `blog-rocm-fp8-gemm-cdna4`'s ladder this single change moved a 4096-cubed
FP8 GEMM from 336.88 to 506.70 TFLOP/s, and it is what makes the subsequent
double-buffering rung work: "the next tile's global-to-LDS transfer runs in
parallel with current MFMA work", synchronized with `s_waitcnt vmcnt(...)`.

**Transfers to RDNA4? No.** gfx1201 has no direct-to-LDS path. On RDNA4 the tile
takes the long route — `global_load_dwordx4` into VGPRs, then `ds_write_b128`
into LDS — and the VGPRs in flight count against occupancy for the duration.
That is a real cost, and it is one reason RDNA4 GEMM kernels favour smaller
K-tiles than their CDNA equivalents.

## The full path

`blog-rocm-memory-scheduling` traces a three-stage pipelined CDNA GEMM end to
end: HBM -> VGPR -> LDS -> AGPR. Per K-tile each lane issues 16
`buffer_load_dwordx4` for 256 B, stages through a 64 KiB LDS region with
`ds_write_b128` / `ds_read_b128` pairs, and feeds MFMA out of AGPRs that never
spill. On RDNA4 the same diagram loses its last arrow: accumulators stay in
ordinary VGPRs.
